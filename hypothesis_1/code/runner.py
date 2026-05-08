from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep
from typing import Any

from hypothesis_1.code.config import ALL_CONDITIONS, build_agent_config
from hypothesis_1.code.paths import (
    APPWORLD_OUTPUTS_DIR,
    DEFAULT_REFLECTION_MANIFEST_PATH,
    OUTPUTS_DIR,
    REPO_ROOT,
    configure_appworld_root,
    ensure_hypothesis_tree,
)
from hypothesis_1.code.schedule import RunSpec, build_run_specs, load_manifest

ALL_CONDITION_SELECTOR = "all"


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def read_existing_record(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        record = json.load(file)
    record["skipped_existing"] = True
    return record


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def import_agent_classes() -> Any:
    import appworld_agents.code.simplified.react_code_agent  # noqa: F401
    import hypothesis_1.code.guided_agent  # noqa: F401
    from appworld_agents.code.simplified.agent import Agent

    return Agent


def make_agent(spec: RunSpec, *, verbose: bool, log_lm_calls: bool) -> Any:
    Agent = import_agent_classes()
    config = build_agent_config(
        seed=spec.seed,
        verbose=verbose,
        color=sys.stdout.isatty(),
        log_lm_calls=log_lm_calls,
        guidance_text=spec.guidance_text,
        guidance_source=spec.reflection_source,
    )
    return Agent.from_dict(config)


def save_prompt_preview(spec: RunSpec, agent: Any) -> None:
    preview_dir = OUTPUTS_DIR / "prompt_previews" / spec.condition
    preview_json = preview_dir / f"{spec.flat_run_id}.json"
    preview_md = preview_dir / f"{spec.flat_run_id}.md"
    write_json(
        {
            "saved_at_utc": datetime.now(UTC).isoformat(),
            "run_spec": spec.to_dict(),
            "messages": agent.messages,
        },
        preview_json,
    )
    preview_md.parent.mkdir(parents=True, exist_ok=True)
    with preview_md.open("w", encoding="utf-8") as file:
        file.write(f"# {spec.condition} / {spec.run_id}\n\n")
        for message in agent.messages:
            file.write(f"## {message['role']}\n\n{message['content']}\n\n")


def run_record_path(spec: RunSpec) -> Path:
    return (
        APPWORLD_OUTPUTS_DIR
        / spec.experiment_name
        / "tasks"
        / spec.task_id
        / "misc"
        / "run_record.json"
    )


def selected_conditions(condition: str) -> list[str]:
    if condition == ALL_CONDITION_SELECTOR:
        return list(ALL_CONDITIONS)
    return [condition]


def build_selected_specs(
    *,
    condition: str,
    manifest: dict[str, Any],
    limit: int | None = None,
    task_id: str | None = None,
    sample_id: str | None = None,
    run_id: str | None = None,
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for condition_name in selected_conditions(condition):
        specs.extend(
            build_run_specs(
                condition_name,
                manifest,
                limit=limit,
                task_id=task_id,
                sample_id=sample_id,
            )
        )
    if run_id is not None:
        specs = [spec for spec in specs if spec.run_id == run_id]
    return specs


def safe_log_name(spec: RunSpec) -> str:
    safe = []
    for character in f"{spec.condition}__{spec.flat_run_id}":
        safe.append(character if character.isalnum() or character in "._=-" else "_")
    return "".join(safe)[:220]


def experiment_name_for(spec: RunSpec, *, dry_run: bool) -> str:
    if dry_run:
        return f"hypothesis_1/dry_runs/{spec.condition}/{spec.run_id}"
    return spec.experiment_name


def run_spec(
    spec: RunSpec,
    *,
    dry_run: bool,
    verbose: bool,
    log_lm_calls: bool,
) -> dict[str, Any]:
    configure_appworld_root()
    from appworld import AppWorld
    from appworld.common.io import maybe_create_parent_directory, write_file
    from appworld.common.path_store import path_store
    from appworld_agents.code.simplified.agent import ExecutionIO

    agent = make_agent(spec, verbose=verbose, log_lm_calls=log_lm_calls)
    agent_config = build_agent_config(
        seed=spec.seed,
        verbose=verbose,
        color=sys.stdout.isatty(),
        log_lm_calls=log_lm_calls,
        guidance_text=spec.guidance_text,
        guidance_source=spec.reflection_source,
    )
    appworld_config = agent_config.get("appworld_config", {})
    experiment_name = experiment_name_for(spec, dry_run=dry_run)
    record: dict[str, Any] = {
        "condition": spec.condition,
        "task_id": spec.task_id,
        "run_id": spec.run_id,
        "sample_id": spec.sample_id,
        "reflection_source": spec.reflection_source,
        "seed": spec.seed,
        "experiment_name": experiment_name,
        "dry_run": dry_run,
        "started_at_utc": datetime.now(UTC).isoformat(),
    }

    agent.logger.initialize(
        experiment_name=experiment_name,
        num_tasks=1,
        num_processes=1,
        process_index=0,
        extra_experiment_info={
            "Condition": spec.condition,
            "Run ID": spec.run_id,
            "Seed": spec.seed,
        },
    )
    try:
        with AppWorld.initializer(
            update_defaults=True,
            experiment_name=experiment_name,
            **appworld_config,
        ):
            with AppWorld(task_id=spec.task_id) as world:
                agent.initialize(world)
                record["appworld_task_output_dir"] = str(
                    Path(path_store.experiment_outputs)
                    / experiment_name
                    / "tasks"
                    / spec.task_id
                )
                if dry_run:
                    save_prompt_preview(spec, agent)
                    record["status"] = "dry_run"
                    record["messages_count"] = len(agent.messages)
                    return record

                execution_outputs: list[ExecutionIO] = []
                run_error: str | None = None
                for _ in range(agent.max_steps):
                    agent.step_number += 1
                    execution_inputs, usage, status = agent.next_execution_inputs_usage_and_status(
                        execution_outputs
                    )
                    if status.failed:
                        run_error = status.message
                        agent.logger.show_message(role="termination", content=status.message)
                        break
                    execution_outputs_raw = world.batch_execute(
                        [execution_input.content for execution_input in execution_inputs]
                    )
                    execution_outputs = [
                        ExecutionIO(content=execution_output, metadata=execution_input.metadata)
                        for execution_input, execution_output in zip(
                            execution_inputs, execution_outputs_raw, strict=True
                        )
                    ]
                    agent.usage_tracker.add(spec.task_id, usage)
                    agent.log_usage()
                    if world.task_completed() or agent.usage_tracker.exceeded(spec.task_id):
                        break

                try:
                    evaluation = world.evaluate(suppress_errors=True).to_dict()
                except Exception as exception:
                    evaluation = {
                        "success": False,
                        "evaluation_error": f"{type(exception).__name__}: {exception}",
                    }
                if run_error:
                    evaluation["runner_error"] = run_error
                record.update(
                    {
                        "status": "completed",
                        "success": bool(evaluation.get("success", False)),
                        "evaluation": evaluation,
                        "steps": agent.step_number,
                        "usage": agent.usage_tracker.task_usage[spec.task_id].dict(),
                        "completed_at_utc": datetime.now(UTC).isoformat(),
                    }
                )
                task_misc_dir = Path(world.output_misc_directory)
                write_json(evaluation, task_misc_dir / "evaluation.json")
                write_json(record, task_misc_dir / "run_record.json")
                write_file(
                    "",
                    os.path.join(
                        path_store.experiment_outputs,
                        experiment_name,
                        "tasks",
                        spec.task_id,
                        "misc",
                        "finished",
                    ),
                )
    except Exception as exception:
        record.update(
            {
                "status": "runner_failed",
                "success": False,
                "runner_error": f"{type(exception).__name__}: {exception}",
                "traceback": traceback.format_exc(),
                "completed_at_utc": datetime.now(UTC).isoformat(),
            }
        )
        record_path = (
            Path(path_store.experiment_outputs)
            / experiment_name
            / "tasks"
            / spec.task_id
            / "misc"
            / "run_record.json"
        )
        maybe_create_parent_directory(str(record_path))
        write_json(record, record_path)
    finally:
        try:
            agent.logger.complete_task()
        except Exception:
            pass
    return record


def read_worker_summary(summary_path: Path, spec: RunSpec, returncode: int) -> dict[str, Any]:
    if summary_path.exists():
        summary = read_json(summary_path)
        runs = summary.get("runs", [])
        if runs:
            record = runs[0]
            record["parallel_worker_returncode"] = returncode
            return record
    record_path = run_record_path(spec)
    if record_path.exists():
        record = read_existing_record(record_path)
        record["parallel_worker_returncode"] = returncode
        return record
    return {
        "condition": spec.condition,
        "task_id": spec.task_id,
        "run_id": spec.run_id,
        "sample_id": spec.sample_id,
        "reflection_source": spec.reflection_source,
        "seed": spec.seed,
        "status": "parallel_worker_failed",
        "success": False,
        "parallel_worker_returncode": returncode,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }


def child_command_for_spec(
    *,
    spec: RunSpec,
    args: argparse.Namespace,
    summary_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "hypothesis_1.code.runner",
        "--manifest",
        str(args.manifest),
        "--condition",
        spec.condition,
        "--run-id",
        spec.run_id,
        "--parallel",
        "1",
        "--summary-output",
        str(summary_path),
    ]
    if args.dry_run:
        command.append("--dry-run")
    if args.rerun_existing:
        command.append("--rerun-existing")
    if args.quiet:
        command.append("--quiet")
    if args.no_lm_log:
        command.append("--no-lm-log")
    return command


def run_specs_parallel(
    specs: list[RunSpec],
    *,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_dir = OUTPUTS_DIR / "parallel_logs" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running {len(specs)} scheduled specs with parallel={args.parallel}")
    print(f"Worker logs: {log_dir}")

    records_by_index: dict[int, dict[str, Any]] = {}
    pending: list[tuple[int, RunSpec]] = []
    for index, spec in enumerate(specs, start=1):
        existing_record_path = run_record_path(spec)
        if existing_record_path.exists() and not args.dry_run and not args.rerun_existing:
            print(f"[{index}/{len(specs)}] skipping existing {spec.condition} {spec.run_id}")
            records_by_index[index] = read_existing_record(existing_record_path)
            continue
        pending.append((index, spec))

    running: dict[subprocess.Popen[str], dict[str, Any]] = {}
    next_pending = 0
    completed = len(records_by_index)
    worker_failures = 0

    def start_next() -> None:
        nonlocal next_pending
        if next_pending >= len(pending):
            return
        index, spec = pending[next_pending]
        next_pending += 1
        log_path = log_dir / f"{index:04d}_{safe_log_name(spec)}.log"
        summary_path = log_dir / f"{index:04d}_{safe_log_name(spec)}.summary.json"
        log_file = log_path.open("w", encoding="utf-8")
        command = child_command_for_spec(spec=spec, args=args, summary_path=summary_path)
        log_file.write(f"# started_at_utc={datetime.now(UTC).isoformat()}\n")
        log_file.write("# command=" + " ".join(command) + "\n\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        running[process] = {
            "index": index,
            "spec": spec,
            "started": monotonic(),
            "log_path": log_path,
            "summary_path": summary_path,
            "log_file": log_file,
        }

    try:
        while next_pending < len(pending) and len(running) < args.parallel:
            start_next()
        while running:
            finished = [process for process in running if process.poll() is not None]
            if not finished:
                sleep(1)
                continue
            for process in finished:
                job = running.pop(process)
                job["log_file"].close()
                index = int(job["index"])
                spec = job["spec"]
                elapsed = monotonic() - float(job["started"])
                returncode = int(process.returncode)
                record = read_worker_summary(Path(job["summary_path"]), spec, returncode)
                record["parallel_log_path"] = str(job["log_path"])
                record["parallel_seconds"] = round(elapsed, 2)
                records_by_index[index] = record
                completed += 1
                if returncode != 0 or record.get("status") == "parallel_worker_failed":
                    worker_failures += 1
                print(
                    f"[{completed}/{len(specs)}] rc={returncode} "
                    f"status={record.get('status')} success={record.get('success')} "
                    f"{spec.condition} {spec.run_id} seconds={elapsed:.1f}"
                )
                while next_pending < len(pending) and len(running) < args.parallel:
                    start_next()
    except KeyboardInterrupt:
        print("Interrupted; terminating active parallel workers.", file=sys.stderr)
        for process, job in running.items():
            try:
                process.terminate()
            finally:
                job["log_file"].close()
        deadline = monotonic() + 10
        for process in running:
            remaining = max(0.0, deadline - monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.kill()
        raise

    records = [records_by_index[index] for index in sorted(records_by_index)]
    if worker_failures:
        print(f"Parallel run completed with {worker_failures} worker process failures.")
    return records


def has_parallel_worker_failures(records: list[dict[str, Any]]) -> bool:
    return any(
        record.get("parallel_worker_returncode", 0) != 0
        or record.get("status") == "parallel_worker_failed"
        for record in records
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hypothesis 1 AppWorld schedules.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_REFLECTION_MANIFEST_PATH)
    parser.add_argument("--condition", choices=(*ALL_CONDITIONS, ALL_CONDITION_SELECTOR), required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run exactly one scheduled run_id after condition/task/sample filtering.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--parallel",
        type=int,
        default=int(os.environ.get("HYPOTHESIS_1_PARALLEL", "1")),
        help="Number of independent run specs to execute concurrently.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=OUTPUTS_DIR / "last_run_summary.json",
        help="Where to write this invocation's summary JSON.",
    )
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Rerun even when this schedule already has a run_record.json.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable verbose AppWorld logs.")
    parser.add_argument("--no-lm-log", action="store_true", help="Do not write lm_calls.jsonl.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.parallel < 1:
        raise SystemExit("--parallel must be at least 1.")
    ensure_hypothesis_tree()
    manifest = load_manifest(args.manifest)
    specs = build_selected_specs(
        condition=args.condition,
        manifest=manifest,
        limit=args.limit,
        task_id=args.task_id,
        sample_id=args.sample_id,
        run_id=args.run_id,
    )
    if not specs:
        raise SystemExit("No runs matched the requested filters.")

    if args.parallel > 1 and len(specs) > 1:
        summary_records = run_specs_parallel(specs, args=args)
    else:
        summary_records = []
        for index, spec in enumerate(specs, start=1):
            print(f"[{index}/{len(specs)}] {spec.condition} {spec.run_id} task={spec.task_id}")
            existing_record_path = run_record_path(spec)
            if existing_record_path.exists() and not args.dry_run and not args.rerun_existing:
                print(f"Skipping existing run record: {existing_record_path}")
                summary_records.append(read_existing_record(existing_record_path))
                continue
            summary_records.append(
                run_spec(
                    spec,
                    dry_run=args.dry_run,
                    verbose=not args.quiet,
                    log_lm_calls=not args.no_lm_log,
                )
            )
    write_json({"runs": summary_records}, args.summary_output)
    print(f"Wrote summary to {args.summary_output}")
    if has_parallel_worker_failures(summary_records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
