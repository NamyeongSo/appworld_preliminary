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
class JobResult:
    response_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class ChatResult:
    response_text: str
    assistant_reasoning: str | None = None
    assistant_reasoning_content: str | None = None
    raw_response: dict[str, Any] | None = None


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    default_manifest_path: Path,
    default_trajectories_dir: Path,
    default_output_dir: Path,
    analysis_dirname: str,
) -> None:
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_manifest_path,
        help="Path to a single counterfactual manifest JSON.",
    )
    parser.add_argument(
        "--trajectories-dir",
        type=Path,
        default=default_trajectories_dir,
        help="Directory containing trajectory JSON files for single-manifest mode.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=None,
        help="Recursively find and evaluate counterfactual manifests under this output root.",
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
        default=analysis_dirname,
        help="Directory name created next to each discovered manifest in batch mode.",
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
        default=default_output_dir,
        help="Directory where analysis artifacts are saved in single-manifest mode.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default=None,
        help="Optional output file stem override.",
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
        help="Skip a job if both output text and metadata files already exist.",
    )
    parser.add_argument(
        "--print-response",
        action="store_true",
        help="Print each model response to stdout after saving.",
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


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def trim_text(value: Any, max_chars: int = MAX_MESSAGE_CHARS) -> Any:
    if not isinstance(value, str) or len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}\n... [truncated {len(value) - max_chars} chars]"


def trim_message_for_prompt(message: dict[str, Any]) -> dict[str, Any]:
    return {
        key: trim_text(value) if key == "content" else value
        for key, value in message.items()
    }


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


def summarize_trajectory(data: dict[str, Any]) -> dict[str, Any]:
    messages = data.get("messages", [])
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
        "evaluation": data.get("evaluation"),
        "error": data.get("error"),
    }


def trajectory_success_value(data: dict[str, Any]) -> bool | None:
    evaluation = data.get("evaluation")
    if isinstance(evaluation, dict):
        success = evaluation.get("success")
        if isinstance(success, bool):
            return success
    return None


def trajectory_outcome_label(data: dict[str, Any]) -> str:
    success = trajectory_success_value(data)
    if success is True:
        return "successful"
    if success is False:
        return "failed"
    return "unknown"


def load_trajectory_file(path: Path) -> dict[str, Any]:
    data = read_json(path)
    messages = data.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"Trajectory file is missing a list-valued 'messages' field: {path}")
    return data


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
    *,
    output_dir: Path,
    output_stem: str,
    metadata: dict[str, Any],
    response_text: str,
) -> JobResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    response_path = output_dir / f"{output_stem}.txt"
    metadata_path = output_dir / f"{output_stem}.json"
    response_path.write_text(response_text, encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return JobResult(response_path=response_path, metadata_path=metadata_path)


def should_skip_output(output_dir: Path, output_stem: str, skip_existing: bool) -> bool:
    if not skip_existing:
        return False
    return (output_dir / f"{output_stem}.txt").is_file() and (output_dir / f"{output_stem}.json").is_file()


async def run_jobs(
    jobs: list[Callable[[], Awaitable[dict[str, Any]]]],
    *,
    concurrency: int,
    print_response: bool,
) -> tuple[int, int]:
    queue: asyncio.Queue[Callable[[], Awaitable[dict[str, Any]]] | None] = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)

    failure_count = 0
    completed_count = 0
    total = len(jobs)
    counter_lock = asyncio.Lock()

    async def worker() -> None:
        nonlocal failure_count
        nonlocal completed_count
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
                print(f"[{current_index}/{total}] Failed: {exc}")
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
    return completed_count, failure_count
