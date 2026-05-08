import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from openai import AsyncOpenAI


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_NAME = os.getenv("VLLM_MODEL", "google/gemma-4-26B-A4B-it")
BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8002/v1")
MAX_TOKENS = int(os.getenv("VLLM_MAX_TOKENS", "8192"))
TEMPERATURE = float(os.getenv("VLLM_TEMPERATURE", "0.2"))
TOP_P = float(os.getenv("VLLM_TOP_P", "0.95"))
TOP_K = int(os.getenv("VLLM_TOP_K", "64"))
ENABLE_THINKING = os.getenv("GEMMA_ENABLE_THINKING", "1") != "0"

DEFAULT_MANIFEST_PATH = Path("counterfactual/manifest.json")
DEFAULT_TRAJECTORIES_DIR = Path("counterfactual/trajectories")
DEFAULT_OUTPUT_DIR = Path("counterfactual/analysis")
DEFAULT_SYSTEM_PROMPT_PATH = SCRIPT_DIR / "reflection_system.txt"
DEFAULT_USER_PROMPT_PATH = SCRIPT_DIR / "reflection_user.txt"
DEFAULT_MANIFEST_GLOB = "**/counterfactual/manifest.json"
MAX_MESSAGES_PER_TRAJECTORY = int(os.getenv("REFLECTION_MAX_MESSAGES_PER_TRAJECTORY", "12"))
MAX_MESSAGE_CHARS = int(os.getenv("REFLECTION_MAX_MESSAGE_CHARS", "3000"))
DEFAULT_CONCURRENCY = int(os.getenv("REFLECTION_CONCURRENCY", "20"))

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "NONE")
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "NONE"),
    base_url=BASE_URL,
)


@dataclass(frozen=True)
class EvaluationTarget:
    manifest_path: Path
    trajectories_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class ResolvedPair:
    success_trajectory_id: str
    failure_trajectory_id: str
    has_true_success: bool = True
    resolution_note: str | None = None


@dataclass(frozen=True)
class JobResult:
    response_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class ChatResult:
    response_text: str
    assistant_reasoning: str | None = None
    assistant_reasoning_content: str | None = None
    raw_response: dict[str, Any] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load final success/failure trajectories from counterfactual manifests, "
            "analyze their messages with Gemma 4, and save reflection results. "
            "Supports both single-manifest and concurrent batch evaluation modes."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to a single counterfactual manifest JSON.",
    )
    parser.add_argument(
        "--trajectories-dir",
        type=Path,
        default=DEFAULT_TRAJECTORIES_DIR,
        help="Directory containing trajectory JSON files for single-manifest mode.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=None,
        help=(
            "Batch mode: recursively find and evaluate every counterfactual manifest "
            "under this output root. Example: experiments/outputs/counterfactual/..."
        ),
    )
    parser.add_argument(
        "--manifest-glob",
        type=str,
        default=DEFAULT_MANIFEST_GLOB,
        help="Glob used in batch mode to discover manifest files under --outputs-root.",
    )
    parser.add_argument(
        "--analysis-dirname",
        type=str,
        default="analysis",
        help=(
            "Directory name created next to each discovered manifest in batch mode. "
            "Ignored in single-manifest mode unless --output-dir is explicitly set."
        ),
    )
    parser.add_argument(
        "--system-prompt",
        type=Path,
        default=DEFAULT_SYSTEM_PROMPT_PATH,
        help="Path to reflection system prompt template.",
    )
    parser.add_argument(
        "--user-prompt",
        type=Path,
        default=DEFAULT_USER_PROMPT_PATH,
        help="Path to reflection user prompt template.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where analysis artifacts are saved in single-manifest mode.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default=None,
        help="Optional output file stem. Defaults to task/case/trajectory ids.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional batch-mode limit on how many manifests to evaluate.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Batch mode: skip an item if both output text and metadata files already exist.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Batch mode: stop immediately on the first failed manifest.",
    )
    parser.add_argument(
        "--print-response",
        action="store_true",
        help="Print the model response to stdout after saving.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the LLM call and only build/save the resolved prompt payload.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Number of concurrent reflections to keep in flight.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_trajectory_path(trajectory_id: str, trajectories_dir: Path) -> Path:
    path = trajectories_dir / f"{trajectory_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory file not found for id {trajectory_id!r}: {path}")
    return path


def load_trajectory(trajectory_id: str, trajectories_dir: Path) -> dict[str, Any]:
    path = resolve_trajectory_path(trajectory_id, trajectories_dir)
    data = read_json(path)
    messages = data.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"Trajectory {trajectory_id!r} is missing a list-valued 'messages' field.")
    return data


def trim_text(value: Any, max_chars: int = MAX_MESSAGE_CHARS) -> Any:
    if not isinstance(value, str) or len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}\n... [truncated {len(value) - max_chars} chars]"


def trim_message_for_prompt(message: dict[str, Any]) -> dict[str, Any]:
    trimmed_message: dict[str, Any] = {}
    for key, value in message.items():
        if key == "content":
            trimmed_message[key] = trim_text(value)
        else:
            trimmed_message[key] = value
    return trimmed_message


def select_messages_for_prompt(messages: list[dict[str, Any]]) -> dict[str, Any]:
    if len(messages) <= MAX_MESSAGES_PER_TRAJECTORY:
        selected_messages = messages
        omitted_count = 0
    else:
        head_count = MAX_MESSAGES_PER_TRAJECTORY // 2
        tail_count = MAX_MESSAGES_PER_TRAJECTORY - head_count
        selected_messages = [*messages[:head_count], *messages[-tail_count:]]
        omitted_count = len(messages) - len(selected_messages)

    return {
        "total_messages": len(messages),
        "selected_messages_count": len(selected_messages),
        "omitted_messages_count": omitted_count,
        "messages": [trim_message_for_prompt(message) for message in selected_messages],
    }


def trajectory_success_value(data: dict[str, Any]) -> bool | None:
    evaluation = data.get("evaluation")
    if isinstance(evaluation, dict):
        success = evaluation.get("success")
        if isinstance(success, bool):
            return success
    return None


def trajectory_pass_count(data: dict[str, Any]) -> int:
    evaluation = data.get("evaluation")
    if not isinstance(evaluation, dict):
        return 0
    passes = evaluation.get("passes")
    if isinstance(passes, list):
        return len(passes)
    return 0


def load_manifest_trajectories(manifest: dict[str, Any], trajectories_dir: Path) -> list[dict[str, Any]]:
    trajectory_ids = manifest.get("trajectory_ids")
    if not isinstance(trajectory_ids, list):
        return []

    trajectories: list[dict[str, Any]] = []
    for trajectory_id in trajectory_ids:
        if not isinstance(trajectory_id, str):
            continue
        path = trajectories_dir / f"{trajectory_id}.json"
        if not path.is_file():
            continue
        trajectories.append(read_json(path))
    return trajectories


def resolve_pair_ids(manifest: dict[str, Any], trajectories_dir: Path) -> ResolvedPair:
    success_trajectory_id = manifest.get("final_success_trajectory_id")
    failure_trajectory_id = manifest.get("final_failure_trajectory_id")
    has_true_success = isinstance(success_trajectory_id, str) and bool(success_trajectory_id)
    notes: list[str] = []

    if has_true_success and isinstance(failure_trajectory_id, str) and failure_trajectory_id:
        return ResolvedPair(
            success_trajectory_id=success_trajectory_id,
            failure_trajectory_id=failure_trajectory_id,
            has_true_success=True,
        )

    trajectories = load_manifest_trajectories(manifest, trajectories_dir)
    if not trajectories:
        raise ValueError("Could not resolve comparison pair because no trajectory files were available.")

    if not (isinstance(failure_trajectory_id, str) and failure_trajectory_id):
        if any(data.get("trajectory_id") == "root" for data in trajectories):
            failure_trajectory_id = "root"
            notes.append("Manifest lacked final_failure_trajectory_id; used root as fallback failure trajectory.")
        else:
            failed_trajectories = [data for data in trajectories if trajectory_success_value(data) is False]
            if failed_trajectories:
                failure_trajectory_id = failed_trajectories[0].get("trajectory_id")
                notes.append("Manifest lacked final_failure_trajectory_id; used the first failed trajectory as fallback.")

    if not (isinstance(success_trajectory_id, str) and success_trajectory_id):
        non_failure_trajectories = [
            data for data in trajectories if data.get("trajectory_id") != failure_trajectory_id
        ]
        successful_trajectories = [data for data in non_failure_trajectories if trajectory_success_value(data) is True]
        replay_met_trajectories = [
            data for data in non_failure_trajectories if data.get("replay_objective_met") is True
        ]
        candidate_pool = successful_trajectories or replay_met_trajectories or non_failure_trajectories
        if candidate_pool:
            candidate_pool = sorted(
                candidate_pool,
                key=lambda data: (trajectory_pass_count(data), data.get("attempt_index") or -1),
                reverse=True,
            )
            success_trajectory_id = candidate_pool[0].get("trajectory_id")
            notes.append(
                "Manifest lacked final_success_trajectory_id; both compared trajectories should be treated as failures, and the first trajectory is only the strongest available fallback comparison target."
            )

    if not (isinstance(success_trajectory_id, str) and success_trajectory_id):
        raise ValueError("Could not resolve a fallback success trajectory id from the manifest trajectories.")
    if not (isinstance(failure_trajectory_id, str) and failure_trajectory_id):
        raise ValueError("Could not resolve a fallback failure trajectory id from the manifest trajectories.")

    return ResolvedPair(
        success_trajectory_id=success_trajectory_id,
        failure_trajectory_id=failure_trajectory_id,
        has_true_success=has_true_success,
        resolution_note=" ".join(notes) if notes else None,
    )


def summarize_trajectory(data: dict[str, Any]) -> dict[str, Any]:
    messages = data.get("messages", [])
    evaluation = data.get("evaluation")
    error = data.get("error")
    return {
        "trajectory_id": data.get("trajectory_id"),
        "task_id": data.get("task_id"),
        "case": data.get("case"),
        "attempt_index": data.get("attempt_index"),
        "parent_trajectory_id": data.get("parent_trajectory_id"),
        "source_trajectory_id": data.get("source_trajectory_id"),
        "replay_objective_met": data.get("replay_objective_met"),
        "action_changed": data.get("action_changed"),
        "is_final_pair_member": data.get("is_final_pair_member"),
        "messages_count": len(messages),
        "steps_count": len(data.get("steps", [])),
        "evaluation": evaluation,
        "error": error,
    }


def build_output_stem(
    manifest: dict[str, Any],
    success_trajectory_id: str,
    failure_trajectory_id: str,
    explicit_stem: str | None,
) -> str:
    if explicit_stem:
        return explicit_stem

    task_id = manifest.get("task_id", "unknown_task")
    case = manifest.get("case", "unknown_case")
    return f"{task_id}_{case}_{success_trajectory_id}_vs_{failure_trajectory_id}"


def render_user_prompt(
    template: str,
    trajectories_dir: Path,
    success_trajectory: dict[str, Any],
    failure_trajectory: dict[str, Any],
    has_true_success: bool,
) -> str:
    base_prompt = template.replace("{{ traces_folder }}", str(trajectories_dir))

    analysis_payload = {
        "success_trajectory": {
            "summary": summarize_trajectory(success_trajectory),
            "selected_messages": select_messages_for_prompt(success_trajectory["messages"]),
        },
        "failure_trajectory": {
            "summary": summarize_trajectory(failure_trajectory),
            "selected_messages": select_messages_for_prompt(failure_trajectory["messages"]),
        },
        "has_true_success": has_true_success,
    }

    resolution_note = success_trajectory.get("_resolution_note")
    resolution_note_block = ""
    if resolution_note:
        resolution_note_block = f"Note: {resolution_note}\n\n"

    if has_true_success:
        pair_description = (
            f"- successful trajectory id: {success_trajectory['trajectory_id']}\n"
            f"- failed trajectory id: {failure_trajectory['trajectory_id']}\n\n"
        )
    else:
        pair_description = (
            f"- fallback comparison trajectory id: {success_trajectory['trajectory_id']}\n"
            f"- failed trajectory id: {failure_trajectory['trajectory_id']}\n\n"
            "Important: this manifest has no true successful trajectory. Treat both trajectories as failures.\n"
            "The first trajectory is only a fallback comparison target, not a success example.\n\n"
        )

    return (
        f"{base_prompt}\n\n"
        "## Manifest-resolved final pair\n"
        "The following trajectories were selected directly from manifest.json or resolved from the manifest when ids were missing:\n"
        f"{pair_description}"
        f"{resolution_note_block}"
        "## Trajectory data to analyze\n"
        "Use the summaries and messages below as the source trajectories for your reflection.\n"
        "The `messages` arrays are the trajectory conversations.\n\n"
        "```json\n"
        f"{json.dumps(analysis_payload, ensure_ascii=False, indent=2)}\n"
        "```"
    )


def _maybe_dump_response(response: Any) -> dict[str, Any] | None:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return None


def _message_field_as_text(message: Any, field_name: str) -> str | None:
    value = getattr(message, field_name, None)
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        joined = "\n".join(part for part in parts if part)
        return joined or None
    return str(value)


async def call_gemma_chat(system_prompt: str, user_prompt: str) -> ChatResult:
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        extra_body={
            "top_k": TOP_K,
            "chat_template_kwargs": {
                "enable_thinking": ENABLE_THINKING,
            },
            "skip_special_tokens": False,
        },
    )
    message = response.choices[0].message
    return ChatResult(
        response_text=message.content or "",
        assistant_reasoning=_message_field_as_text(message, "reasoning"),
        assistant_reasoning_content=_message_field_as_text(message, "reasoning_content"),
        raw_response=_maybe_dump_response(response),
    )


def save_outputs(
    output_dir: Path,
    output_stem: str,
    manifest_path: Path,
    manifest: dict[str, Any],
    success_trajectory: dict[str, Any],
    failure_trajectory: dict[str, Any],
    system_prompt_path: Path,
    user_prompt_path: Path,
    system_prompt: str,
    user_prompt: str,
    chat_result: ChatResult,
) -> JobResult:
    output_dir.mkdir(parents=True, exist_ok=True)

    response_path = output_dir / f"{output_stem}.txt"
    metadata_path = output_dir / f"{output_stem}.json"

    response_path.write_text(chat_result.response_text, encoding="utf-8")

    metadata = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": MODEL_NAME,
        "base_url": BASE_URL,
        "manifest_path": str(manifest_path),
        "manifest": {
            "task_id": manifest.get("task_id"),
            "case": manifest.get("case"),
            "final_success_trajectory_id": manifest.get("final_success_trajectory_id"),
            "final_failure_trajectory_id": manifest.get("final_failure_trajectory_id"),
        },
        "success_trajectory_summary": summarize_trajectory(success_trajectory),
        "failure_trajectory_summary": summarize_trajectory(failure_trajectory),
        "system_prompt_path": str(system_prompt_path),
        "user_prompt_path": str(user_prompt_path),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "response_path": str(response_path),
        "response_text": chat_result.response_text,
        "assistant_reasoning": chat_result.assistant_reasoning,
        "assistant_reasoning_content": chat_result.assistant_reasoning_content,
        "raw_response": chat_result.raw_response,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return JobResult(response_path=response_path, metadata_path=metadata_path)


def discover_targets(args: argparse.Namespace) -> list[EvaluationTarget]:
    if args.outputs_root is None:
        return [
            EvaluationTarget(
                manifest_path=args.manifest,
                trajectories_dir=args.trajectories_dir,
                output_dir=args.output_dir,
            )
        ]

    manifests = sorted(args.outputs_root.glob(args.manifest_glob))
    if args.limit is not None:
        manifests = manifests[: args.limit]

    return [
        EvaluationTarget(
            manifest_path=manifest_path,
            trajectories_dir=manifest_path.parent / "trajectories",
            output_dir=manifest_path.parent / args.analysis_dirname,
        )
        for manifest_path in manifests
    ]


def should_skip_target(target: EvaluationTarget, output_stem: str, skip_existing: bool) -> bool:
    if not skip_existing:
        return False
    response_path = target.output_dir / f"{output_stem}.txt"
    metadata_path = target.output_dir / f"{output_stem}.json"
    return response_path.is_file() and metadata_path.is_file()


async def evaluate_target(
    target: EvaluationTarget,
    args: argparse.Namespace,
    system_prompt: str,
    user_prompt_template: str,
) -> dict[str, Any]:
    manifest = read_json(target.manifest_path)
    resolved_pair = resolve_pair_ids(manifest, target.trajectories_dir)
    success_trajectory_id = resolved_pair.success_trajectory_id
    failure_trajectory_id = resolved_pair.failure_trajectory_id

    success_trajectory = load_trajectory(success_trajectory_id, target.trajectories_dir)
    failure_trajectory = load_trajectory(failure_trajectory_id, target.trajectories_dir)
    if resolved_pair.resolution_note:
        success_trajectory = {**success_trajectory, "_resolution_note": resolved_pair.resolution_note}

    user_prompt = render_user_prompt(
        user_prompt_template,
        target.trajectories_dir,
        success_trajectory,
        failure_trajectory,
        resolved_pair.has_true_success,
    )

    if args.dry_run:
        chat_result = ChatResult(
            response_text="<guidance>\n"
            "Dry run only: the prompt was assembled successfully, but no LLM call was made.\n"
            "</guidance>"
        )
    else:
        chat_result = await call_gemma_chat(system_prompt, user_prompt)

    output_stem = build_output_stem(
        manifest,
        success_trajectory_id,
        failure_trajectory_id,
        args.output_stem,
    )
    result = save_outputs(
        target.output_dir,
        output_stem,
        target.manifest_path,
        manifest,
        success_trajectory,
        failure_trajectory,
        args.system_prompt,
        args.user_prompt,
        system_prompt,
        user_prompt,
        chat_result,
    )

    return {
        "manifest_path": str(target.manifest_path),
        "output_dir": str(target.output_dir),
        "output_stem": output_stem,
        "response_path": str(result.response_path),
        "metadata_path": str(result.metadata_path),
        "task_id": manifest.get("task_id"),
        "case": manifest.get("case"),
    }


async def run_jobs(
    jobs: list[Callable[[], Awaitable[dict[str, Any]]]],
    *,
    concurrency: int,
    print_response: bool,
    fail_fast: bool,
) -> tuple[int, int]:
    queue: asyncio.Queue[Callable[[], Awaitable[dict[str, Any]]] | None] = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)

    failure_count = 0
    completed_count = 0
    total = len(jobs)
    counter_lock = asyncio.Lock()
    first_exception: Exception | None = None

    async def worker() -> None:
        nonlocal failure_count
        nonlocal completed_count
        nonlocal first_exception
        while True:
            job = await queue.get()
            if job is None:
                queue.task_done()
                return

            try:
                result = await job()
            except Exception as exc:
                async with counter_lock:
                    failure_count += 1
                    current_index = completed_count + failure_count
                    if fail_fast and first_exception is None:
                        first_exception = exc
                print(f"[{current_index}/{total}] Failed: {exc}")
                if fail_fast:
                    while not queue.empty():
                        try:
                            pending = queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        if pending is not None:
                            queue.task_done()
            else:
                async with counter_lock:
                    completed_count += 1
                    current_index = completed_count + failure_count
                print(f"[{current_index}/{total}] Saved analysis text to: {result['response_path']}")
                print(f"[{current_index}/{total}] Saved analysis metadata to: {result['metadata_path']}")
                if print_response:
                    print("\n=== MODEL RESPONSE ===\n")
                    print(read_text(Path(result["response_path"])))
            finally:
                queue.task_done()

    worker_count = max(1, concurrency)
    workers = [asyncio.create_task(worker()) for _ in range(worker_count)]
    await queue.join()
    for _ in workers:
        queue.put_nowait(None)
    await asyncio.gather(*workers)

    if first_exception is not None:
        raise first_exception

    return completed_count, failure_count


async def main_async() -> None:
    args = parse_args()
    system_prompt = read_text(args.system_prompt)
    user_prompt_template = read_text(args.user_prompt)
    targets = discover_targets(args)

    if not targets:
        search_root = args.outputs_root if args.outputs_root is not None else args.manifest.parent
        raise FileNotFoundError(
            f"No counterfactual manifests found under {search_root} using glob {args.manifest_glob!r}."
        )

    total_targets = len(targets)
    skipped = 0
    jobs: list[Callable[[], Awaitable[dict[str, Any]]]] = []

    for index, target in enumerate(targets, start=1):
        manifest = read_json(target.manifest_path)
        resolved_pair = resolve_pair_ids(manifest, target.trajectories_dir)
        output_stem = build_output_stem(
            manifest,
            resolved_pair.success_trajectory_id,
            resolved_pair.failure_trajectory_id,
            args.output_stem,
        )

        if should_skip_target(target, output_stem, args.skip_existing):
            skipped += 1
            print(f"[{index}/{total_targets}] Skipped existing: {target.manifest_path}")
            continue

        jobs.append(
            lambda current_target=target: evaluate_target(
                current_target,
                args,
                system_prompt,
                user_prompt_template,
            )
        )

    completed, failed = await run_jobs(
        jobs,
        concurrency=args.concurrency,
        print_response=args.print_response,
        fail_fast=args.fail_fast,
    )

    print(
        "\nBatch summary: "
        f"total={total_targets}, completed={completed}, skipped={skipped}, failed={failed}"
    )

    if failed:
        raise SystemExit(1)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
