import argparse
import json
import re
from pathlib import Path
from typing import Any

from datasets import Dataset


DEFAULT_OUTPUTS_ROOT = Path("experiments/outputs/counterfactual/google/gemma-4-26b-a4b-it")
DEFAULT_TASKS_ROOT = Path("data/tasks")
DEFAULT_DATASETS_ROOT = Path("experiments/datasets/reflection_hf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Hugging Face reflection datasets from counterfactual analysis outputs."
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=DEFAULT_OUTPUTS_ROOT,
        help="Root directory containing counterfactual output chunks.",
    )
    parser.add_argument(
        "--tasks-root",
        type=Path,
        default=DEFAULT_TASKS_ROOT,
        help="Root directory containing task specs.json files.",
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=DEFAULT_DATASETS_ROOT,
        help="Root directory where Hugging Face datasets will be saved.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_task_query(task_id: str, tasks_root: Path) -> str:
    specs_path = tasks_root / task_id / "specs.json"
    if not specs_path.is_file():
        raise FileNotFoundError(f"Task specs not found: {specs_path}")
    return str(read_json(specs_path)["instruction"])


def extract_guidance_text(response_text: str) -> str:
    stripped = response_text.strip()
    if stripped.startswith("<guidance>") and stripped.endswith("</guidance>"):
        return stripped
    return stripped


def extract_reasoning_text(response_text: str) -> str | None:
    patterns = [
        r"<thinking>(.*?)</thinking>",
        r"<reasoning>(.*?)</reasoning>",
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip() or None
    return None


def build_label_with_reasoning(response_text: str, reasoning_text: str | None) -> str:
    if not reasoning_text:
        return response_text
    stripped = response_text.lstrip()
    if stripped.startswith("<thinking>") or stripped.startswith("<reasoning>"):
        return response_text
    return f"<thinking>\n{reasoning_text.strip()}\n</thinking>\n\n{response_text}"


def is_dry_run_response(response_text: str) -> bool:
    return "Dry run only" in response_text


def detect_source_type(path: Path) -> str:
    if "analysis_root" in path.parts:
        return "root"
    if "analysis_random_pairs" in path.parts:
        return "random_pair"
    if "analysis" in path.parts:
        return "pair"
    raise ValueError(f"Could not detect source type from path: {path}")


def build_messages(task_query: str, reflection_text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You generate task-specific reflections that help an agent solve the task successfully.",
        },
        {
            "role": "user",
            "content": (
                f"Task query:\n{task_query}\n\n"
                "Generate a concise reflection in <guidance> tags."
            ),
        },
        {
            "role": "assistant",
            "content": reflection_text,
        },
    ]


def build_actual_messages(system_prompt: str, user_prompt: str, assistant_label: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_label},
    ]


def pair_labels_from_record(record: dict[str, Any], source_type: str) -> tuple[str | None, str | None]:
    if source_type == "pair":
        success_eval = record.get("success_trajectory_summary", {}).get("evaluation", {})
        failure_eval = record.get("failure_trajectory_summary", {}).get("evaluation", {})
        return (
            "successful" if success_eval.get("success") is True else "unknown",
            "failed" if failure_eval.get("success") is False else "unknown",
        )
    if source_type == "root":
        success = record.get("root_trajectory_summary", {}).get("evaluation", {}).get("success")
        label = "successful" if success is True else "failed" if success is False else "unknown"
        return (label, None)
    if source_type == "random_pair":
        a_success = record.get("trajectory_a_summary", {}).get("evaluation", {}).get("success")
        b_success = record.get("trajectory_b_summary", {}).get("evaluation", {}).get("success")
        label_a = "successful" if a_success is True else "failed" if a_success is False else "unknown"
        label_b = "successful" if b_success is True else "failed" if b_success is False else "unknown"
        return (label_a, label_b)
    return (None, None)


def build_record(path: Path, tasks_root: Path) -> dict[str, Any] | None:
    data = read_json(path)
    response_text = str(data.get("response_text", "")).strip()
    if not response_text or is_dry_run_response(response_text):
        return None

    source_type = detect_source_type(path)
    manifest = data.get("manifest", {})
    task_id = str(manifest.get("task_id"))
    case = str(manifest.get("case"))
    task_query = read_task_query(task_id, tasks_root)
    system_prompt = str(data.get("system_prompt", ""))
    user_prompt = str(data.get("user_prompt", ""))
    assistant_label = response_text
    reflection_text = extract_guidance_text(response_text)
    reasoning_text = (
        data.get("assistant_reasoning_content")
        or data.get("assistant_reasoning")
        or extract_reasoning_text(response_text)
    )
    assistant_label_with_reasoning = build_label_with_reasoning(assistant_label, reasoning_text)
    label_a, label_b = pair_labels_from_record(data, source_type)

    sample_id = f"{task_id}::{case}::{source_type}::{path.stem}"
    return {
        "sample_id": sample_id,
        "task_id": task_id,
        "case": case,
        "task_query": task_query,
        "reflection_text": reflection_text,
        "source_type": source_type,
        "source_path": str(path),
        "manifest_path": str(data.get("manifest_path", "")),
        "outcome_label_a": label_a,
        "outcome_label_b": label_b,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "assistant_label": assistant_label,
        "assistant_label_with_reasoning": assistant_label_with_reasoning,
        "assistant_reasoning": reasoning_text,
        "messages": build_actual_messages(system_prompt, user_prompt, assistant_label),
        "messages_with_reasoning": build_actual_messages(
            system_prompt,
            user_prompt,
            assistant_label_with_reasoning,
        ),
        "messages_query_only": build_messages(task_query, reflection_text),
    }


def collect_records(outputs_root: Path, tasks_root: Path, source_glob: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(outputs_root.glob(source_glob)):
        record = build_record(path, tasks_root)
        if record is not None:
            records.append(record)
    return records


def save_dataset(records: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_list(records)
    dataset.save_to_disk(str(output_dir))


def write_preview_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    model_name = args.outputs_root.name

    pair_records = collect_records(args.outputs_root, args.tasks_root, "**/analysis/*.json")
    root_records = collect_records(args.outputs_root, args.tasks_root, "**/analysis_root/*.json")
    random_records = collect_records(args.outputs_root, args.tasks_root, "**/analysis_random_pairs/*.json")
    all_records = [*pair_records, *root_records, *random_records]

    pair_dir = args.datasets_root / f"{model_name}_pair_only"
    all_dir = args.datasets_root / f"{model_name}_all_inclusive"

    save_dataset(pair_records, pair_dir)
    save_dataset(all_records, all_dir)

    write_preview_jsonl(pair_records, pair_dir.with_suffix(".jsonl"))
    write_preview_jsonl(all_records, all_dir.with_suffix(".jsonl"))

    print(
        json.dumps(
            {
                "pair_only_count": len(pair_records),
                "all_inclusive_count": len(all_records),
                "pair_only_dir": str(pair_dir),
                "all_inclusive_dir": str(all_dir),
                "pair_only_jsonl": str(pair_dir.with_suffix(".jsonl")),
                "all_inclusive_jsonl": str(all_dir.with_suffix(".jsonl")),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
