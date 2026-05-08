import argparse
import asyncio
import json
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
DEFAULT_OUTPUT_DIR = Path("counterfactual/analysis_root")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate reflections from root trajectories only. "
            "Supports single-manifest and batch discovery modes."
        )
    )
    add_common_args(
        parser,
        default_manifest_path=DEFAULT_MANIFEST_PATH,
        default_trajectories_dir=DEFAULT_TRAJECTORIES_DIR,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        analysis_dirname="analysis_root",
    )
    return parser.parse_args()


def build_output_stem(manifest: dict[str, Any], explicit_stem: str | None) -> str:
    if explicit_stem:
        return explicit_stem
    return f"{manifest.get('task_id', 'unknown_task')}_{manifest.get('case', 'unknown_case')}_root"


def render_user_prompt(template: str, trajectories_dir: Path, root_trajectory: dict[str, Any]) -> str:
    base_prompt = template.replace("{{ traces_folder }}", str(trajectories_dir))
    root_outcome_label = trajectory_outcome_label(root_trajectory)
    analysis_payload = {
        "root_trajectory": {
            "outcome_label": root_outcome_label,
            "summary": summarize_trajectory(root_trajectory),
            "selected_messages": select_messages_for_prompt(root_trajectory["messages"]),
        }
    }
    return (
        f"{base_prompt}\n\n"
        "## Single root trajectory mode\n"
        "Only the root trajectory is provided. There is no contrastive pair for this sample.\n"
        "Infer useful task guidance from the root trajectory alone: identify the attempted strategy,\n"
        "likely pitfalls, missing recovery behavior, and improvements that would help future runs.\n\n"
        f"- root trajectory id: {root_trajectory['trajectory_id']}\n\n"
        f"- root trajectory outcome label: {root_outcome_label}\n\n"
        "## Trajectory data to analyze\n"
        "Use the summary and messages below as the source trajectory for your reflection.\n\n"
        "```json\n"
        f"{json.dumps(analysis_payload, ensure_ascii=False, indent=2)}\n"
        "```"
    )


async def evaluate_target(
    *,
    target: EvaluationTarget,
    args: argparse.Namespace,
    system_prompt: str,
    user_prompt_template: str,
) -> dict[str, Any]:
    manifest = read_json(target.manifest_path)
    output_stem = build_output_stem(manifest, args.output_stem)
    if should_skip_output(target.output_dir, output_stem, args.skip_existing):
        return {
            "response_path": str(target.output_dir / f"{output_stem}.txt"),
            "metadata_path": str(target.output_dir / f"{output_stem}.json"),
        }

    root_path = target.trajectories_dir / "root.json"
    if not root_path.is_file():
        raise FileNotFoundError(f"Missing root trajectory: {root_path}")

    root_trajectory = load_trajectory_file(root_path)
    user_prompt = render_user_prompt(user_prompt_template, target.trajectories_dir, root_trajectory)
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
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": MODEL_NAME,
            "base_url": BASE_URL,
            "mode": "root_only",
            "manifest_path": str(target.manifest_path),
            "trajectories_dir": str(target.trajectories_dir),
            "root_trajectory_path": str(root_path),
            "manifest": {
                "task_id": manifest.get("task_id"),
                "case": manifest.get("case"),
            },
            "root_trajectory_summary": summarize_trajectory(root_trajectory),
            "system_prompt_path": str(args.system_prompt),
            "user_prompt_path": str(args.user_prompt),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
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

    jobs = [
        (
            lambda target=target: evaluate_target(
                target=target,
                args=args,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
            )
        )
        for target in targets
    ]
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
