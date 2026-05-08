from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from hypothesis_1.code.config import ALL_CONDITIONS, REFLECTION_CONDITION_TO_SOURCE
from hypothesis_1.code.paths import (
    APPWORLD_OUTPUTS_DIR,
    DEFAULT_METRICS_PATH,
    DEFAULT_REFLECTION_MANIFEST_PATH,
    DEFAULT_REPORT_PATH,
    ensure_hypothesis_tree,
)
from hypothesis_1.code.schedule import build_run_specs, load_manifest, unique_task_ids


DEFAULT_PASS10_SUBSET_METRICS_PATH = DEFAULT_METRICS_PATH.with_name(
    "hypothesis_1_pass10_task_subset_metrics.json"
)
DEFAULT_PASS10_SUBSET_REPORT_PATH = DEFAULT_REPORT_PATH.with_name(
    "hypothesis_1_pass10_task_subset_report.md"
)
DEFAULT_FIGURES_DIR = DEFAULT_REPORT_PATH.parent / "figures"
NORMALIZED_RUNS_PER_TASK = 5


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    phat = successes / total
    denominator = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def success_rate(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return mean(1.0 if record.get("success") else 0.0 for record in records)


def token_total(record: dict[str, Any]) -> int | float:
    tokens = record.get("usage", {}).get("tokens", {})
    return sum(tokens.get(key, 0) for key in ("input_cache_miss", "input_cache_hit", "output"))


def collect_run_records(outputs_root: Path = APPWORLD_OUTPUTS_DIR) -> list[dict[str, Any]]:
    records = []
    root = outputs_root / "hypothesis_1"
    for path in sorted(root.glob("**/tasks/*/misc/run_record.json")):
        record = read_json(path)
        record["_record_path"] = str(path)
        if "success" not in record:
            record["success"] = bool(record.get("evaluation", {}).get("success", False))
        records.append(record)
    return records


def grouped_by(records: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get(key))].append(record)
    return dict(groups)


def condition_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition = grouped_by(records, "condition")
    metrics: dict[str, Any] = {}
    for condition, condition_records in sorted(by_condition.items()):
        successes = sum(1 for record in condition_records if record.get("success"))
        total = len(condition_records)
        ci_low, ci_high = wilson_interval(successes, total)
        task_rates = {
            task_id: success_rate(task_records)
            for task_id, task_records in grouped_by(condition_records, "task_id").items()
        }
        metrics[condition] = {
            "attempts": total,
            "successes": successes,
            "success_rate": successes / total if total else 0.0,
            "success_rate_ci95": [ci_low, ci_high],
            "num_tasks": len(task_rates),
            "macro_by_task_success": mean(task_rates.values()) if task_rates else 0.0,
            "mean_steps": mean(record.get("steps", 0) for record in condition_records)
            if condition_records
            else 0.0,
            "max_steps": max(record.get("steps", 0) for record in condition_records)
            if condition_records
            else 0,
            "mean_tokens": mean(token_total(record) for record in condition_records)
            if condition_records
            else 0.0,
            "max_tokens": max(token_total(record) for record in condition_records)
            if condition_records
            else 0,
            "task_success_rates": task_rates,
        }
    return metrics


def normalized_record_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("seed", 0),
        record.get("manifest_index", 0),
        record.get("sample_id") or "",
        record.get("run_id") or "",
    )


def records_normalized_to_five_runs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_records: list[dict[str, Any]] = []
    for condition, condition_records in grouped_by(records, "condition").items():
        for _, task_records in grouped_by(condition_records, "task_id").items():
            sorted_records = sorted(task_records, key=normalized_record_sort_key)
            if condition == "vanilla_react_pass1":
                if sorted_records:
                    normalized_records.extend([sorted_records[0]] * NORMALIZED_RUNS_PER_TASK)
                continue
            normalized_records.extend(sorted_records[:NORMALIZED_RUNS_PER_TASK])
    return normalized_records


def normalized_condition_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_records = records_normalized_to_five_runs(records)
    metrics = condition_metrics(normalized_records)
    for condition_data in metrics.values():
        condition_data["normalization"] = {
            "runs_per_task": NORMALIZED_RUNS_PER_TASK,
            "vanilla_react_pass1": "single seed repeated to five equivalent attempts",
            "vanilla_react_pass10": "first five seeds per completed task",
            "other_conditions": "first five available attempts per task",
        }
    return metrics


def pass_at_10_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    pass10_records = [
        record for record in records if record.get("condition") == "vanilla_react_pass10"
    ]
    task_groups = grouped_by(pass10_records, "task_id")
    task_passes = {
        task_id: any(record.get("success") for record in task_records)
        for task_id, task_records in task_groups.items()
    }
    successes = sum(1 for value in task_passes.values() if value)
    total = len(task_passes)
    ci_low, ci_high = wilson_interval(successes, total)
    return {
        "tasks": total,
        "task_successes": successes,
        "pass_at_10": successes / total if total else 0.0,
        "pass_at_10_ci95": [ci_low, ci_high],
        "task_passes": task_passes,
    }


def paired_task_deltas(metrics: dict[str, Any], pass10: dict[str, Any]) -> dict[str, Any]:
    vanilla_pass1 = metrics.get("vanilla_react_pass1", {}).get("task_success_rates", {})
    vanilla_pass10 = {
        task_id: 1.0 if passed else 0.0 for task_id, passed in pass10.get("task_passes", {}).items()
    }
    deltas: dict[str, Any] = {}
    for condition in REFLECTION_CONDITION_TO_SOURCE:
        reflection_rates = metrics.get(condition, {}).get("task_success_rates", {})
        rows = []
        for task_id, reflection_rate in sorted(reflection_rates.items()):
            row = {
                "task_id": task_id,
                "reflection_rate": reflection_rate,
                "delta_vs_vanilla_pass1": None,
                "delta_vs_vanilla_pass10": None,
            }
            if task_id in vanilla_pass1:
                row["delta_vs_vanilla_pass1"] = reflection_rate - vanilla_pass1[task_id]
            if task_id in vanilla_pass10:
                row["delta_vs_vanilla_pass10"] = reflection_rate - vanilla_pass10[task_id]
            rows.append(row)
        deltas[condition] = {
            "rows": rows,
            "mean_delta_vs_vanilla_pass1": mean(
                row["delta_vs_vanilla_pass1"]
                for row in rows
                if row["delta_vs_vanilla_pass1"] is not None
            )
            if any(row["delta_vs_vanilla_pass1"] is not None for row in rows)
            else None,
            "mean_delta_vs_vanilla_pass10": mean(
                row["delta_vs_vanilla_pass10"]
                for row in rows
                if row["delta_vs_vanilla_pass10"] is not None
            )
            if any(row["delta_vs_vanilla_pass10"] is not None for row in rows)
            else None,
        }
    return deltas


def expected_run_status(
    *,
    manifest_path: Path,
    records: list[dict[str, Any]],
    conditions: list[str],
    task_ids: set[str] | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    present = {(record.get("condition"), record.get("run_id")) for record in records}
    missing: list[dict[str, Any]] = []
    duplicate_keys = []
    seen: set[tuple[Any, Any]] = set()
    for record in records:
        key = (record.get("condition"), record.get("run_id"))
        if key in seen:
            duplicate_keys.append({"condition": key[0], "run_id": key[1]})
        seen.add(key)
    for condition in conditions:
        for spec in build_run_specs(condition, manifest):
            if task_ids is not None and spec.task_id not in task_ids:
                continue
            if (spec.condition, spec.run_id) not in present:
                missing.append(spec.to_dict())
    return {
        "expected_conditions": conditions,
        "expected_task_ids": sorted(task_ids) if task_ids is not None else None,
        "missing_runs": missing,
        "num_missing_runs": len(missing),
        "duplicate_run_keys": duplicate_keys,
        "num_duplicate_run_keys": len(duplicate_keys),
    }


def build_metrics(
    *,
    records: list[dict[str, Any]],
    manifest_path: Path | None = None,
    expected_conditions: list[str] | None = None,
) -> dict[str, Any]:
    metrics = condition_metrics(records)
    pass10 = pass_at_10_metrics(records)
    result: dict[str, Any] = {
        "num_records": len(records),
        "conditions": metrics,
        "normalized_5run_conditions": normalized_condition_metrics(records),
        "vanilla_react_pass10": pass10,
        "paired_task_deltas": paired_task_deltas(metrics, pass10),
    }
    if manifest_path is not None and expected_conditions:
        result["expected_run_status"] = expected_run_status(
            manifest_path=manifest_path,
            records=records,
            conditions=expected_conditions,
        )
    return result


def pass10_task_ids(records: list[dict[str, Any]]) -> set[str]:
    return {
        str(record.get("task_id"))
        for record in records
        if record.get("condition") == "vanilla_react_pass10"
    }


def build_pass10_task_subset_metrics(
    *,
    records: list[dict[str, Any]],
    manifest_path: Path,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    included_task_ids = pass10_task_ids(records)
    all_task_ids = set(unique_task_ids(manifest))
    subset_records = [
        record for record in records if str(record.get("task_id")) in included_task_ids
    ]
    metrics = build_metrics(records=subset_records)
    metrics["subset"] = {
        "name": "vanilla_react_pass10_task_subset",
        "description": (
            "Only tasks with at least one completed vanilla_react_pass10 run are included. "
            "All conditions are restricted to that same task support before comparison."
        ),
        "included_task_ids": sorted(included_task_ids),
        "excluded_task_ids": sorted(all_task_ids - included_task_ids),
        "num_included_tasks": len(included_task_ids),
        "num_excluded_tasks": len(all_task_ids - included_task_ids),
    }
    metrics["expected_run_status"] = expected_run_status(
        manifest_path=manifest_path,
        records=subset_records,
        conditions=list(ALL_CONDITIONS),
        task_ids=included_task_ids,
    )
    return metrics


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100 * value:.1f}%"


def format_condition_table(metrics: dict[str, Any]) -> str:
    condition_metrics_ = metrics.get("normalized_5run_conditions") or metrics.get("conditions", {})
    lines = [
        (
            "| Condition | Attempts | Tasks | Success Rate | Macro By Task | "
            "Mean Steps | Max Steps | Mean Tokens |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in ALL_CONDITIONS:
        data = condition_metrics_.get(condition)
        if not data:
            lines.append(f"| `{condition}` | 0 | 0 | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{condition}`",
                    str(data["attempts"]),
                    str(data["num_tasks"]),
                    pct(data["success_rate"]),
                    pct(data["macro_by_task_success"]),
                    f"{data['mean_steps']:.1f}",
                    f"{data['max_steps']:.0f}",
                    f"{data['mean_tokens']:.0f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def conclusion(metrics: dict[str, Any]) -> str:
    expected_status = metrics.get("expected_run_status", {})
    missing_runs = expected_status.get("num_missing_runs", 0)
    if missing_runs:
        return (
            "Conclusion: Partial-run result only. The completed records support the hypothesis "
            "so far because all reflection-guided pass@1 variants underperform observed vanilla "
            "ReAct pass@10 on macro-by-task success, but the final conclusion should wait for "
            f"the remaining {missing_runs} scheduled runs."
        )
    pass10_value = metrics.get("vanilla_react_pass10", {}).get("pass_at_10", 0.0)
    reflection_values = [
        metrics.get("conditions", {}).get(condition, {}).get("macro_by_task_success")
        for condition in REFLECTION_CONDITION_TO_SOURCE
    ]
    if not pass10_value or any(value is None for value in reflection_values):
        return "Conclusion: Inconclusive until all scheduled conditions have completed."
    if all(float(value) < pass10_value for value in reflection_values if value is not None):
        return (
            "Conclusion: Supported for this run set. All reflection-guided pass@1 variants "
            "underperform vanilla ReAct pass@10 on macro-by-task success."
        )
    return (
        "Conclusion: Not supported as stated for this run set. At least one reflection-guided "
        "pass@1 variant matches or exceeds vanilla ReAct pass@10 on macro-by-task success."
    )


def report_title(metrics: dict[str, Any]) -> str:
    if metrics.get("subset", {}).get("name") == "vanilla_react_pass10_task_subset":
        return "Hypothesis 1 Pass@10 Task Subset Report"
    return "Hypothesis 1 Report"


def safe_filename_part(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)


CONDITION_COLORS = {
    "reflection_analysis_pass1": "#4C78A8",
    "reflection_random_pairs_pass1": "#F58518",
    "reflection_random_3_pass1": "#FF9DA6",
    "reflection_random_4_pass1": "#B279A2",
    "reflection_random_5_pass1": "#9D755D",
    "reflection_random_6_pass1": "#BAB0AC",
    "reflection_random_7_pass1": "#F1CE63",
    "reflection_root_pass1": "#54A24B",
    "vanilla_react_pass1": "#E45756",
    "vanilla_react_pass10": "#72B7B2",
}


def histogram_paths_for(report_path: Path) -> tuple[Path, Path]:
    stem = report_path.stem
    figures_dir = report_path.parent / "figures"
    return (
        figures_dir / f"{stem}_steps_by_condition_distribution.png",
        figures_dir / f"{stem}_tokens_by_condition_distribution.png",
    )


def write_distribution_plots_by_condition(
    records: list[dict[str, Any]], report_path: Path
) -> dict[str, Path]:
    if not records:
        return {}
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def normal_pdf(value: float, *, mean_: float, stddev: float) -> float:
        return (
            math.exp(-0.5 * ((value - mean_) / stddev) ** 2)
            / (stddev * math.sqrt(2 * math.pi))
        )

    def save_metric_plot(
        path: Path,
        *,
        metric_name: str,
        xlabel: str,
        value_for_record: Any,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        condition_to_values = {
            condition: [
                float(value_for_record(record) or 0)
                for record in condition_records
                if value_for_record(record) is not None
            ]
            for condition, condition_records in grouped_by(records, "condition").items()
        }
        all_values = [value for values in condition_to_values.values() for value in values]
        if not all_values:
            return
        min_value = min(all_values)
        max_value = max(all_values)
        if min_value == max_value:
            max_value = min_value + 1
        bin_count = min(30, max(8, int(math.sqrt(len(all_values)))))
        bin_width = (max_value - min_value) / bin_count
        bins = [min_value + bin_width * index for index in range(bin_count + 1)]
        x_values = [
            min_value + (max_value - min_value) * index / 250
            for index in range(251)
        ]
        active_conditions = [
            condition for condition in ALL_CONDITIONS if condition_to_values.get(condition)
        ]
        grouped_bar_width = bin_width / (len(active_conditions) + 1)
        bin_left_edges = bins[:-1]

        figure, axis = plt.subplots(figsize=(10, 5.5))
        for condition_index, condition in enumerate(active_conditions):
            values = condition_to_values.get(condition, [])
            color = CONDITION_COLORS.get(condition, "#4C78A8")
            counts = []
            for bin_index, left_edge in enumerate(bin_left_edges):
                right_edge = bins[bin_index + 1]
                if bin_index == len(bin_left_edges) - 1:
                    count = sum(left_edge <= value <= right_edge for value in values)
                else:
                    count = sum(left_edge <= value < right_edge for value in values)
                counts.append(count)
            bar_positions = [
                left_edge + grouped_bar_width * (condition_index + 0.5)
                for left_edge in bin_left_edges
            ]
            axis.bar(
                bar_positions,
                counts,
                width=grouped_bar_width * 0.92,
                color=color,
                edgecolor="white",
                align="edge",
                label=f"{condition} histogram",
            )
            if len(values) >= 2:
                mean_value = mean(values)
                variance = mean((value - mean_value) ** 2 for value in values)
                stddev = math.sqrt(variance)
                if stddev > 0:
                    y_values = [
                        normal_pdf(value, mean_=mean_value, stddev=stddev)
                        * len(values)
                        * bin_width
                        for value in x_values
                    ]
                    axis.plot(
                        x_values,
                        y_values,
                        color=color,
                        linewidth=2,
                        label=f"{condition} normal fit",
                    )
                    peak_y = normal_pdf(mean_value, mean_=mean_value, stddev=stddev) * len(
                        values
                    ) * bin_width
                    axis.axvline(
                        mean_value,
                        color=color,
                        linestyle="--",
                        linewidth=1,
                        alpha=0.7,
                    )
                    axis.text(
                        mean_value,
                        peak_y,
                        f"μ={mean_value:.1f}",
                        color=color,
                        fontsize=7,
                        rotation=90,
                        ha="right",
                        va="bottom",
                    )
        axis.set_title(f"{metric_name} Distribution by Condition")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Run count")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)

    steps_path, tokens_path = histogram_paths_for(report_path)
    save_metric_plot(
        steps_path,
        metric_name="Step Count",
        xlabel="Steps",
        value_for_record=lambda record: record.get("steps", 0),
    )
    save_metric_plot(
        tokens_path,
        metric_name="Token Count",
        xlabel="Tokens",
        value_for_record=token_total,
    )
    return {"steps": steps_path, "tokens": tokens_path}


def relative_markdown_path(path: Path, *, from_path: Path) -> str:
    try:
        return path.relative_to(from_path.parent).as_posix()
    except ValueError:
        return path.as_posix()


def write_report(
    metrics: dict[str, Any],
    report_path: Path,
    *,
    histogram_paths: dict[str, Path] | None = None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pass10 = metrics.get("vanilla_react_pass10", {})
    lines = [
        f"# {report_title(metrics)}",
        "",
        f"Raw run records: {metrics['num_records']}",
    ]
    if "subset" in metrics:
        subset = metrics["subset"]
        lines.extend(
            [
                "",
                "## Subset",
                "",
                subset["description"],
                "",
                f"- Included tasks: {subset['num_included_tasks']}",
                f"- Excluded tasks: {subset['num_excluded_tasks']}",
                f"- Excluded task IDs: {', '.join(subset['excluded_task_ids']) or 'none'}",
            ]
        )
    lines.extend(
        [
            "",
            "## 5-Run Normalized Condition Metrics",
            "",
            (
                "`vanilla_react_pass10` uses the first five seeds per completed task. "
                "`vanilla_react_pass1` repeats its single seed result to five equivalent "
                "attempts per task. Reflection conditions use the first five available "
                "attempts per task."
            ),
            "",
            format_condition_table(metrics),
            "",
            "## Vanilla ReAct Pass@10",
            "",
            f"- Tasks: {pass10.get('tasks', 0)}",
            f"- Task successes: {pass10.get('task_successes', 0)}",
            f"- pass@10: {pct(pass10.get('pass_at_10', 0.0))}",
            "",
            "## Histograms",
            "",
        ]
    )
    if histogram_paths is None:
        lines.append("Histogram images were not generated.")
    else:
        steps_path = relative_markdown_path(histogram_paths["steps"], from_path=report_path)
        tokens_path = relative_markdown_path(histogram_paths["tokens"], from_path=report_path)
        lines.extend(
            [
                (
                    "Each figure shows side-by-side condition-specific histograms with "
                    "same-color normal-distribution fits. Dashed vertical lines mark each "
                    "condition mean."
                ),
                "",
                f"![Step Count Distribution by Condition]({steps_path})",
                "",
                f"![Token Count Distribution by Condition]({tokens_path})",
            ]
        )
    lines.extend(
        [
            "",
            "## Paired Task Deltas",
            "",
        ]
    )
    for condition, data in metrics.get("paired_task_deltas", {}).items():
        lines.append(
            f"- `{condition}` mean delta vs vanilla pass@1: "
            f"{pct(data.get('mean_delta_vs_vanilla_pass1'))}; "
            f"vs vanilla pass@10: {pct(data.get('mean_delta_vs_vanilla_pass10'))}"
        )
    if "expected_run_status" in metrics:
        status = metrics["expected_run_status"]
        lines.extend(
            [
                "",
                "## Run Completeness",
                "",
                f"- Missing expected runs: {status['num_missing_runs']}",
                f"- Duplicate run keys: {status['num_duplicate_run_keys']}",
            ]
        )
    lines.extend(["", conclusion(metrics), ""])
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate hypothesis 1 run records.")
    parser.add_argument("--outputs-root", type=Path, default=APPWORLD_OUTPUTS_DIR)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--pass10-subset-metrics-output",
        type=Path,
        default=DEFAULT_PASS10_SUBSET_METRICS_PATH,
    )
    parser.add_argument(
        "--pass10-subset-report-output",
        type=Path,
        default=DEFAULT_PASS10_SUBSET_REPORT_PATH,
    )
    parser.add_argument(
        "--skip-pass10-subset-report",
        action="store_true",
        help="Do not write the same-task-support report restricted to pass@10 tasks.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_REFLECTION_MANIFEST_PATH)
    parser.add_argument(
        "--verify-expected",
        action="store_true",
        help="Compare run records against every scheduled run for all conditions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_hypothesis_tree()
    records = collect_run_records(args.outputs_root)
    metrics = build_metrics(
        records=records,
        manifest_path=args.manifest if args.verify_expected else None,
        expected_conditions=list(ALL_CONDITIONS) if args.verify_expected else None,
    )
    normalized_records = records_normalized_to_five_runs(records)
    histogram_paths = write_distribution_plots_by_condition(
        normalized_records,
        args.report_output,
    )
    write_json(metrics, args.metrics_output)
    write_report(metrics, args.report_output, histogram_paths=histogram_paths)
    print(f"Wrote metrics to {args.metrics_output}")
    print(f"Wrote report to {args.report_output}")
    if not args.skip_pass10_subset_report:
        subset_task_ids = pass10_task_ids(records)
        subset_records = [
            record for record in records if str(record.get("task_id")) in subset_task_ids
        ]
        subset_normalized_records = records_normalized_to_five_runs(subset_records)
        subset_metrics = build_pass10_task_subset_metrics(
            records=records,
            manifest_path=args.manifest,
        )
        subset_histogram_paths = write_distribution_plots_by_condition(
            subset_normalized_records,
            args.pass10_subset_report_output,
        )
        write_json(subset_metrics, args.pass10_subset_metrics_output)
        write_report(
            subset_metrics,
            args.pass10_subset_report_output,
            histogram_paths=subset_histogram_paths,
        )
        print(f"Wrote pass@10 task subset metrics to {args.pass10_subset_metrics_output}")
        print(f"Wrote pass@10 task subset report to {args.pass10_subset_report_output}")


if __name__ == "__main__":
    main()
