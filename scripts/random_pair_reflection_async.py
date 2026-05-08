import argparse
import asyncio
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reflection_async_common import (
    BASE_URL,
    ChatResult,
    EvaluationTarget,
    MODEL_NAME,
    add_common_args,
    call_gemma_chat,
    discover_targets,
    load_trajectory_file,
    read_json,
    read_text,
    run_jobs,
    save_outputs,
    select_messages_for_prompt,
    should_skip_output,
    summarize_trajectory,
    trajectory_outcome_label,
)


DEFAULT_MANIFEST_PATH = Path("counterfactual/manifest.json")
DEFAULT_TRAJECTORIES_DIR = Path("counterfactual/trajectories")
DEFAULT_OUTPUT_DIR = Path("counterfactual/analysis_random_pairs")
DEFAULT_ANALYSIS_DIRNAME = "analysis_random_pairs"


def default_analysis_dirname(sample_size: int) -> str:
    if sample_size == 2:
        return DEFAULT_ANALYSIS_DIRNAME
    return f"analysis_random_{sample_size}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate reflections from randomly selected trajectory groups in each trajectories folder. "
            "Supports single-manifest and batch discovery modes."
        )
    )
    add_common_args(
        parser,
        default_manifest_path=DEFAULT_MANIFEST_PATH,
        default_trajectories_dir=DEFAULT_TRAJECTORIES_DIR,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        analysis_dirname=DEFAULT_ANALYSIS_DIRNAME,
    )
    parser.add_argument(
        "--pairs-per-target",
        type=int,
        default=1,
        help=(
            "How many random trajectory groups to sample per trajectories directory. "
            "Kept as the legacy option name for compatibility."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2,
        help=(
            "How many trajectories to sample for each reflection group. If fewer trajectories "
            "are available, all available trajectories are used."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible pair selection.",
    )
    args = parser.parse_args()
    normalize_args(args, argv=sys.argv[1:])
    return args


def normalize_args(args: argparse.Namespace, *, argv: list[str]) -> None:
    if args.sample_size < 2:
        raise ValueError(f"--sample-size must be at least 2, got {args.sample_size}")
    if args.pairs_per_target < 1:
        raise ValueError(f"--pairs-per-target must be at least 1, got {args.pairs_per_target}")

    if args.sample_size == 2:
        return

    analysis_dirname = default_analysis_dirname(args.sample_size)
    if "--analysis-dirname" not in argv and args.analysis_dirname == DEFAULT_ANALYSIS_DIRNAME:
        args.analysis_dirname = analysis_dirname
    if "--output-dir" not in argv and args.output_dir == DEFAULT_OUTPUT_DIR:
        args.output_dir = DEFAULT_OUTPUT_DIR.parent / analysis_dirname


def list_trajectory_paths(trajectories_dir: Path) -> list[Path]:
    return sorted(path for path in trajectories_dir.glob("*.json") if path.is_file())


def sample_trajectory_groups(
    trajectory_paths: list[Path],
    *,
    count: int,
    sample_size: int,
    rng: random.Random,
) -> list[tuple[Path, ...]]:
    if len(trajectory_paths) < 2:
        raise ValueError(
            f"Need at least 2 trajectory files to sample a reflection group: {trajectory_paths}"
        )
    if sample_size < 2:
        raise ValueError(f"sample_size must be at least 2, got {sample_size}")

    effective_sample_size = min(sample_size, len(trajectory_paths))
    groups: list[tuple[Path, ...]] = []
    for _ in range(count):
        groups.append(tuple(rng.sample(trajectory_paths, effective_sample_size)))
    return groups


def sample_pairs(
    trajectory_paths: list[Path], *, count: int, rng: random.Random
) -> list[tuple[Path, Path]]:
    groups = sample_trajectory_groups(
        trajectory_paths,
        count=count,
        sample_size=2,
        rng=rng,
    )
    return [(group[0], group[1]) for group in groups]


def build_output_stem(
    manifest: dict[str, Any],
    trajectory_ids: list[str],
    group_index: int,
    explicit_stem: str | None,
) -> str:
    if explicit_stem:
        if group_index == 0:
            return explicit_stem
        return f"{explicit_stem}_{group_index + 1:03d}"
    task_id = manifest.get("task_id", "unknown_task")
    case = manifest.get("case", "unknown_case")
    joined_trajectory_ids = "_vs_".join(trajectory_ids)
    return (
        f"{task_id}_{case}_random{len(trajectory_ids)}_"
        f"{group_index + 1:03d}_{joined_trajectory_ids}"
    )


def trajectory_payload(trajectory: dict[str, Any]) -> dict[str, Any]:
    return {
        "trajectory_id": trajectory.get("trajectory_id"),
        "outcome_label": trajectory_outcome_label(trajectory),
        "summary": summarize_trajectory(trajectory),
        "selected_messages": select_messages_for_prompt(trajectory["messages"]),
    }


def render_user_prompt(
    template: str,
    trajectories_dir: Path,
    trajectories: list[dict[str, Any]],
) -> str:
    base_prompt = template.replace("{{ traces_folder }}", str(trajectories_dir))
    analysis_payload = {
        "trajectories": [
            {
                "sample_index": index,
                **trajectory_payload(trajectory),
            }
            for index, trajectory in enumerate(trajectories, start=1)
        ],
    }
    sampled_lines = "\n".join(
        (
            f"- trajectory {index}: {trajectory['trajectory_id']} "
            f"({trajectory_outcome_label(trajectory)})"
        )
        for index, trajectory in enumerate(trajectories, start=1)
    )
    return (
        f"{base_prompt}\n\n"
        "## Randomly sampled trajectory group\n"
        "The following trajectories were sampled uniformly without replacement from the "
        "trajectories folder. If the requested sample size exceeded the number of "
        "available trajectories, the group contains every available trajectory.\n"
        "Use this sampled group as the comparison set for your reflection.\n\n"
        f"{sampled_lines}\n\n"
        "## Trajectory data to analyze\n"
        "Use the summaries and messages below as the source trajectories for your reflection.\n\n"
        "```json\n"
        f"{json.dumps(analysis_payload, ensure_ascii=False, indent=2)}\n"
        "```"
    )


async def evaluate_group(
    *,
    target: EvaluationTarget,
    args: argparse.Namespace,
    system_prompt: str,
    user_prompt_template: str,
    group_paths: tuple[Path, ...],
    group_index: int,
) -> dict[str, Any]:
    manifest = read_json(target.manifest_path)
    trajectories = [load_trajectory_file(path) for path in group_paths]
    trajectory_ids = [
        str(trajectory.get("trajectory_id") or path.stem)
        for trajectory, path in zip(trajectories, group_paths, strict=True)
    ]
    output_stem = build_output_stem(
        manifest,
        trajectory_ids,
        group_index,
        args.output_stem,
    )
    if should_skip_output(target.output_dir, output_stem, args.skip_existing):
        return {
            "response_path": str(target.output_dir / f"{output_stem}.txt"),
            "metadata_path": str(target.output_dir / f"{output_stem}.json"),
        }

    user_prompt = render_user_prompt(
        user_prompt_template,
        target.trajectories_dir,
        trajectories,
    )
    metadata = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": MODEL_NAME,
        "base_url": BASE_URL,
        "mode": "random_pair" if len(group_paths) == 2 else "random_trajectory_group",
        "manifest_path": str(target.manifest_path),
        "trajectories_dir": str(target.trajectories_dir),
        "group_index": group_index,
        "requested_sample_size": args.sample_size,
        "effective_sample_size": len(group_paths),
        "group_paths": [str(path) for path in group_paths],
        "group_trajectory_ids": trajectory_ids,
        "manifest": {
            "task_id": manifest.get("task_id"),
            "case": manifest.get("case"),
        },
        "trajectory_summaries": [
            summarize_trajectory(trajectory) for trajectory in trajectories
        ],
        "system_prompt_path": str(args.system_prompt),
        "user_prompt_path": str(args.user_prompt),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }
    if len(group_paths) == 2:
        metadata.update(
            {
                "pair_index": group_index,
                "pair_paths": [str(path) for path in group_paths],
                "pair_trajectory_ids": trajectory_ids,
                "trajectory_a_summary": summarize_trajectory(trajectories[0]),
                "trajectory_b_summary": summarize_trajectory(trajectories[1]),
            }
        )
    chat_result = (
        ChatResult(
            response_text="<guidance>\nDry run only: the prompt was assembled successfully, but no LLM call was made.\n</guidance>"
        )
        if args.dry_run
        else await call_gemma_chat(system_prompt, user_prompt)
    )

    result = save_outputs(
        output_dir=target.output_dir,
        output_stem=output_stem,
        response_text=chat_result.response_text,
        metadata={
            **metadata,
            "response_text": chat_result.response_text,
            "assistant_reasoning": chat_result.assistant_reasoning,
            "assistant_reasoning_content": chat_result.assistant_reasoning_content,
            "raw_response": chat_result.raw_response,
        },
    )
    return {
        "response_path": str(result.response_path),
        "metadata_path": str(result.metadata_path),
    }


async def evaluate_pair(
    *,
    target: EvaluationTarget,
    args: argparse.Namespace,
    system_prompt: str,
    user_prompt_template: str,
    pair_paths: tuple[Path, Path],
    pair_index: int,
) -> dict[str, Any]:
    return await evaluate_group(
        target=target,
        args=args,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        group_paths=pair_paths,
        group_index=pair_index,
    )


async def main() -> None:
    args = parse_args()
    system_prompt = read_text(args.system_prompt)
    user_prompt_template = read_text(args.user_prompt)
    targets = discover_targets(args)
    if not targets:
        search_root = args.outputs_root if args.outputs_root is not None else args.manifest.parent
        raise FileNotFoundError(
            f"No counterfactual manifests found under {search_root} using glob {args.manifest_glob!r}."
        )

    rng = random.Random(args.seed)
    jobs = []
    for target in targets:
        trajectory_paths = list_trajectory_paths(target.trajectories_dir)
        group_paths_list = sample_trajectory_groups(
            trajectory_paths,
            count=args.pairs_per_target,
            sample_size=args.sample_size,
            rng=rng,
        )
        for group_index, group_paths in enumerate(group_paths_list):
            jobs.append(
                lambda target=target, group_paths=group_paths, group_index=group_index: evaluate_group(
                    target=target,
                    args=args,
                    system_prompt=system_prompt,
                    user_prompt_template=user_prompt_template,
                    group_paths=group_paths,
                    group_index=group_index,
                )
            )

    completed, failed = await run_jobs(
        jobs,
        concurrency=args.concurrency,
        print_response=args.print_response,
    )
    print(f"\nBatch summary: total={len(jobs)}, completed={completed}, failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
