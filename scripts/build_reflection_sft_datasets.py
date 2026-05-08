import argparse
import json
from pathlib import Path

from datasets import Dataset, load_from_disk


DEFAULT_REFLECTION_DATASETS_ROOT = Path("experiments/datasets/reflection_hf")
DEFAULT_SFT_DATASETS_ROOT = Path("experiments/datasets/reflection_sft")
DEFAULT_MODEL_NAME = "gemma-4-26b-a4b-it"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SFT-ready Hugging Face datasets from reflection HF datasets."
    )
    parser.add_argument(
        "--reflection-datasets-root",
        type=Path,
        default=DEFAULT_REFLECTION_DATASETS_ROOT,
        help="Root containing reflection_hf datasets.",
    )
    parser.add_argument(
        "--sft-datasets-root",
        type=Path,
        default=DEFAULT_SFT_DATASETS_ROOT,
        help="Root where SFT datasets will be saved.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Model directory stem used in dataset names.",
    )
    return parser.parse_args()


def convert_dataset(dataset) -> list[dict]:
    rows: list[dict] = []
    for row in dataset:
        rows.append(
            {
                "sample_id": row["sample_id"],
                "task_id": row["task_id"],
                "case": row["case"],
                "source_type": row["source_type"],
                "task_query": row["task_query"],
                "source_path": row["source_path"],
                "manifest_path": row["manifest_path"],
                "outcome_label_a": row["outcome_label_a"],
                "outcome_label_b": row["outcome_label_b"],
                "system_prompt": row["system_prompt"],
                "user_prompt": row["user_prompt"],
                "assistant_label": row["assistant_label_with_reasoning"],
                "assistant_reasoning": row["assistant_reasoning"],
                "reflection_text": row["reflection_text"],
                "messages": row["messages_with_reasoning"],
            }
        )
    return rows


def save_dataset(records: list[dict], output_dir: Path) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).save_to_disk(str(output_dir))


def save_jsonl(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    pair_src = args.reflection_datasets_root / f"{args.model_name}_pair_only"
    all_src = args.reflection_datasets_root / f"{args.model_name}_all_inclusive"

    pair_records = convert_dataset(load_from_disk(str(pair_src)))
    all_records = convert_dataset(load_from_disk(str(all_src)))

    pair_dst = args.sft_datasets_root / f"{args.model_name}_pair_only_sft"
    all_dst = args.sft_datasets_root / f"{args.model_name}_all_inclusive_sft"

    save_dataset(pair_records, pair_dst)
    save_dataset(all_records, all_dst)
    save_jsonl(pair_records, pair_dst.with_suffix(".jsonl"))
    save_jsonl(all_records, all_dst.with_suffix(".jsonl"))

    print(
        json.dumps(
            {
                "pair_only_sft_count": len(pair_records),
                "all_inclusive_sft_count": len(all_records),
                "pair_only_sft_dir": str(pair_dst),
                "all_inclusive_sft_dir": str(all_dst),
                "pair_only_sft_jsonl": str(pair_dst.with_suffix(".jsonl")),
                "all_inclusive_sft_jsonl": str(all_dst.with_suffix(".jsonl")),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
