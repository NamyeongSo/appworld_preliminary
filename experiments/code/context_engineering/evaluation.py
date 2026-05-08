from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ABLATION_SETTINGS = (
    "no_context",
    "no_ratio_filtering",
    "no_defer",
    "no_curriculum",
    "no_counterfactual_replay",
)


@dataclass(frozen=True)
class RunMetrics:
    num_runs: int
    num_tasks: int
    success_count: int
    successful_task_count: int
    success_rate: float
    task_success_rate: float
    total_cost: float
    cost_per_success: float | None
    average_steps_per_task: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_runs": self.num_runs,
            "num_tasks": self.num_tasks,
            "success_count": self.success_count,
            "successful_task_count": self.successful_task_count,
            "success_rate": self.success_rate,
            "task_success_rate": self.task_success_rate,
            "total_cost": self.total_cost,
            "cost_per_success": self.cost_per_success,
            "average_steps_per_task": self.average_steps_per_task,
        }


def build_phase10_evaluation(
    *,
    experiment_name: str,
    dataset_name: str,
    loop_index: int,
    task_ids: list[str],
    trajectories_dir: Path,
    context_injected_runs_dir: Path,
    policy_context_path: Path,
    ratio_estimates_path: Path,
    ratio_metrics_path: Path,
    deferred_queue_path: Path,
    manifest_path: Path,
    ablation_settings: tuple[str, ...] = DEFAULT_ABLATION_SETTINGS,
) -> dict[str, Any]:
    baseline_payloads = _read_trajectory_payloads(trajectories_dir)
    context_payloads = _read_trajectory_payloads(context_injected_runs_dir)
    baseline_metrics = _run_metrics(baseline_payloads)
    context_metrics = _run_metrics(context_payloads)
    ratio_payloads = _read_jsonl_if_exists(ratio_estimates_path)
    ratio_metrics_payload = _read_json_if_exists(ratio_metrics_path)
    ratio_filter_metrics = ratio_metrics_payload.get("phase_6_filtering", {})
    selected_count = int(ratio_filter_metrics.get("selected_count", 0))
    deferred_count = int(ratio_filter_metrics.get("deferred_count", 0))
    discarded_count = int(ratio_filter_metrics.get("discarded_count", 0))
    num_estimation_errors = int(ratio_metrics_payload.get("num_estimation_errors", 0))
    pending_deferred_records = _read_jsonl_if_exists(deferred_queue_path)
    curriculum = _read_json_if_exists(manifest_path).get("phase_9_curriculum", {})
    num_context_items = len(_read_jsonl_if_exists(policy_context_path))
    comparison_available = baseline_metrics.num_runs > 0 and context_metrics.num_runs > 0
    comparison = {
        "available": comparison_available,
        "delta_success_rate": (
            _round_float(context_metrics.success_rate - baseline_metrics.success_rate)
            if comparison_available
            else None
        ),
        "delta_task_success_rate": (
            _round_float(context_metrics.task_success_rate - baseline_metrics.task_success_rate)
            if comparison_available
            else None
        ),
        "delta_cost_per_success": _optional_delta(
            context_metrics.cost_per_success,
            baseline_metrics.cost_per_success,
        ),
        "delta_average_steps_per_task": (
            _round_float(
                context_metrics.average_steps_per_task - baseline_metrics.average_steps_per_task
            )
            if comparison_available
            else None
        ),
    }
    ablation_results = _ablation_results(
        ablation_settings=ablation_settings,
        baseline_metrics=baseline_metrics,
        context_metrics=context_metrics,
        num_ratio_records=len(ratio_payloads),
        selected_count=selected_count,
        deferred_count=deferred_count,
        discarded_count=discarded_count,
        curriculum=curriculum,
    )
    return {
        "schema_version": 1,
        "phase": "phase_10_evaluation_ablation",
        "experiment_name": experiment_name,
        "dataset": dataset_name,
        "loop_index": loop_index,
        "task_ids": task_ids,
        "num_tasks": len(task_ids),
        "baseline": baseline_metrics.to_dict(),
        "context_injected": context_metrics.to_dict(),
        "comparison": comparison,
        "evaluation_warnings": _evaluation_warnings(
            baseline_metrics=baseline_metrics,
            context_metrics=context_metrics,
            num_context_items=num_context_items,
            selected_count=selected_count,
            num_estimation_errors=num_estimation_errors,
        ),
        "success_rate": context_metrics.success_rate,
        "delta_success_rate": comparison["delta_success_rate"],
        "num_context_items": num_context_items,
        "selected_count": selected_count,
        "deferred_count": deferred_count,
        "discarded_count": discarded_count,
        "cost_per_success": context_metrics.cost_per_success,
        "average_steps_per_task": context_metrics.average_steps_per_task,
        "deferred_queue": {
            "pending_count": len(pending_deferred_records),
            "average_defer_round": _average_defer_round(pending_deferred_records),
            "max_defer_round": _max_defer_round(pending_deferred_records),
        },
        "ablation_settings": list(ablation_settings),
        "ablation_results": ablation_results,
    }


def build_evaluation_summary(
    *,
    experiment_name: str,
    dataset_name: str,
    loop_evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    if not loop_evaluations:
        final_evaluation: dict[str, Any] = {}
    else:
        final_evaluation = loop_evaluations[-1]
    return {
        "schema_version": 1,
        "phase": "phase_10_evaluation_summary",
        "experiment_name": experiment_name,
        "dataset": dataset_name,
        "num_loops_evaluated": len(loop_evaluations),
        "loop_indices": [evaluation.get("loop_index") for evaluation in loop_evaluations],
        "final_success_rate": final_evaluation.get("success_rate"),
        "final_delta_success_rate": final_evaluation.get("delta_success_rate"),
        "final_num_context_items": final_evaluation.get("num_context_items"),
        "loop_evaluations": loop_evaluations,
    }


def _ablation_results(
    *,
    ablation_settings: tuple[str, ...],
    baseline_metrics: RunMetrics,
    context_metrics: RunMetrics,
    num_ratio_records: int,
    selected_count: int,
    deferred_count: int,
    discarded_count: int,
    curriculum: Any,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for setting in ablation_settings:
        if setting == "no_context":
            results[setting] = {
                "proxy_success_rate": baseline_metrics.success_rate,
                "delta_vs_context": _round_float(
                    context_metrics.success_rate - baseline_metrics.success_rate
                ),
                "description": "Use baseline trajectories as the no-context proxy.",
            }
        elif setting == "no_ratio_filtering":
            results[setting] = {
                "proxy_selected_count": num_ratio_records,
                "actual_selected_count": selected_count,
                "additional_samples_if_unfiltered": max(0, num_ratio_records - selected_count),
                "description": "Treat every ratio estimate as selected.",
            }
        elif setting == "no_defer":
            results[setting] = {
                "proxy_deferred_count": 0,
                "proxy_discarded_count": discarded_count + deferred_count,
                "actual_deferred_count": deferred_count,
                "description": "Move high-ratio deferred samples to discarded.",
            }
        elif setting == "no_curriculum":
            input_task_ids = curriculum.get("input_task_ids", [])
            curriculum_task_ids = curriculum.get("curriculum_task_ids", [])
            results[setting] = {
                "proxy_task_count": len(input_task_ids),
                "actual_curriculum_task_count": len(curriculum_task_ids),
                "curriculum_filtered_task_count": max(
                    0, len(input_task_ids) - len(curriculum_task_ids)
                ),
                "description": "Use all input tasks without difficulty filtering/order.",
            }
        elif setting == "no_counterfactual_replay":
            results[setting] = {
                "proxy_delta_records": 0,
                "proxy_ratio_records": 0,
                "proxy_insights": 0,
                "description": "Disable counterfactual-derived training signal.",
            }
        else:
            results[setting] = {"description": "Unknown ablation setting."}
    return results


def _evaluation_warnings(
    *,
    baseline_metrics: RunMetrics,
    context_metrics: RunMetrics,
    num_context_items: int,
    selected_count: int,
    num_estimation_errors: int,
) -> list[str]:
    warnings: list[str] = []
    if baseline_metrics.num_runs == 0:
        warnings.append("missing_baseline_runs")
    if context_metrics.num_runs == 0:
        warnings.append("missing_context_injected_runs")
    if num_context_items == 0:
        warnings.append("no_policy_context_items")
    if selected_count == 0:
        warnings.append("no_selected_samples")
    if num_estimation_errors:
        warnings.append("ratio_estimation_errors_present")
    return warnings


def _run_metrics(trajectory_payloads: list[dict[str, Any]]) -> RunMetrics:
    num_runs = len(trajectory_payloads)
    task_ids = sorted({str(payload.get("task_id", "")) for payload in trajectory_payloads})
    success_count = sum(1 for payload in trajectory_payloads if _trajectory_success(payload))
    successful_task_ids = {
        str(payload.get("task_id", ""))
        for payload in trajectory_payloads
        if _trajectory_success(payload)
    }
    total_cost = _round_float(sum(_trajectory_cost(payload) for payload in trajectory_payloads))
    total_steps = sum(_trajectory_steps(payload) for payload in trajectory_payloads)
    return RunMetrics(
        num_runs=num_runs,
        num_tasks=len(task_ids),
        success_count=success_count,
        successful_task_count=len(successful_task_ids),
        success_rate=_safe_rate(success_count, num_runs),
        task_success_rate=_safe_rate(len(successful_task_ids), len(task_ids)),
        total_cost=total_cost,
        cost_per_success=(_round_float(total_cost / success_count) if success_count else None),
        average_steps_per_task=_safe_rate(total_steps, num_runs),
    )


def _read_trajectory_payloads(directory: Path) -> list[dict[str, Any]]:
    if not directory.exists():
        return []
    payloads: list[dict[str, Any]] = []
    for file_path in sorted(directory.glob("*.json")):
        if file_path.name == "manifest.json":
            continue
        payloads.append(_read_json(file_path))
    return payloads


def _trajectory_success(payload: dict[str, Any]) -> bool:
    if "success" in payload:
        return bool(payload["success"])
    evaluation = payload.get("evaluation")
    return bool(evaluation.get("success")) if isinstance(evaluation, dict) else False


def _trajectory_steps(payload: dict[str, Any]) -> int:
    if isinstance(payload.get("num_steps"), int):
        return int(payload["num_steps"])
    steps = payload.get("steps")
    return len(steps) if isinstance(steps, list) else 0


def _trajectory_cost(payload: dict[str, Any]) -> float:
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return 0.0
    return sum(_usage_cost(step.get("usage")) for step in steps if isinstance(step, dict))


def _usage_cost(usage: Any) -> float:
    if not isinstance(usage, dict):
        return 0.0
    cost = usage.get("cost")
    if isinstance(cost, int | float):
        return float(cost)
    if isinstance(cost, dict):
        return float(sum(value for value in cost.values() if isinstance(value, int | float)))
    return 0.0


def _average_defer_round(records: list[dict[str, Any]]) -> float:
    rounds = [_optional_int(record.get("defer_round")) for record in records]
    rounds = [round_ for round_ in rounds if round_ is not None]
    return _safe_rate(sum(rounds), len(rounds))


def _max_defer_round(records: list[dict[str, Any]]) -> int:
    rounds = [_optional_int(record.get("defer_round")) for record in records]
    rounds = [round_ for round_ in rounds if round_ is not None]
    return max(rounds) if rounds else 0


def _optional_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return _round_float(value - baseline)


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return _round_float(float(numerator) / float(denominator))


def _round_float(value: float) -> float:
    return round(float(value), 10)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_json(file_path: Path) -> dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _read_json_if_exists(file_path: Path) -> dict[str, Any]:
    if not file_path.exists():
        return {}
    return _read_json(file_path)


def _read_jsonl_if_exists(file_path: Path) -> list[dict[str, Any]]:
    if not file_path.exists():
        return []
    with file_path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
