from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from appworld_agents.code.context_engineering.config import (
    ContextEngineeringConfig,
    load_context_engineering_config,
)
from appworld_agents.code.context_engineering.counterfactual import (
    CounterfactualPairRecord,
    CounterfactualReplayExecutor,
    CounterfactualTrajectoryRecord,
    ExistingCounterfactualReplayExecutor,
)
from appworld_agents.code.context_engineering.delta import (
    DeltaCfRecord,
    extract_delta_cf_records,
)
from appworld_agents.code.context_engineering.evaluation import (
    build_evaluation_summary,
    build_phase10_evaluation,
)
from appworld_agents.code.context_engineering.filter import (
    RatioFilterDecisionRecord,
    filter_ratio_records,
)
from appworld_agents.code.context_engineering.insight import (
    HeuristicInsightExtractor,
    InsightExtractor,
    InsightRecord,
    extract_insight_records,
)
from appworld_agents.code.context_engineering.policy_context import (
    PolicyContextItem,
    build_policy_context_items,
    estimate_token_count,
    inject_policy_context_prompt_template,
    merge_policy_context_items,
    render_policy_context_prompt,
)
from appworld_agents.code.context_engineering.ratio import (
    ActionSampleRecord,
    AppWorldRatioActionSampler,
    RatioActionSampler,
    RatioEstimateRecord,
    estimate_ratio_from_samples,
    messages_from_state_snapshot,
)
from appworld_agents.code.context_engineering.trajectory import (
    AppWorldBaselineTrajectoryExecutor,
    BaselineTrajectoryExecutor,
    BaselineTrajectoryRecord,
)


@dataclass(frozen=True)
class ContextEngineeringOutputPaths:
    experiment_root: Path
    loop_dir: Path
    trajectories_dir: Path
    counterfactual_dir: Path
    difficulty_manifest_path: Path
    task_id_to_success_rate_path: Path
    task_id_to_difficulty_path: Path
    selected_samples_path: Path
    deferred_queue_path: Path
    global_deferred_queue_path: Path
    discarded_samples_path: Path
    delta_cf_records_path: Path
    ratio_metrics_path: Path
    insights_path: Path
    insight_manifest_path: Path
    policy_context_path: Path
    context_prompt_path: Path
    context_injected_prompt_path: Path
    context_injected_agent_config_path: Path
    context_injected_runs_dir: Path
    evaluation_metrics_path: Path
    ablation_manifest_path: Path
    evaluation_summary_path: Path
    metrics_path: Path
    config_path: Path
    manifest_path: Path

    def ensure_directories(self) -> None:
        for directory in (
            self.experiment_root,
            self.loop_dir,
            self.trajectories_dir,
            self.counterfactual_dir,
            self.context_injected_runs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        for file_path in self._file_paths():
            file_path.parent.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, str]:
        return {field_name: str(value) for field_name, value in self.__dict__.items()}

    def _file_paths(self) -> tuple[Path, ...]:
        return (
            self.difficulty_manifest_path,
            self.task_id_to_success_rate_path,
            self.task_id_to_difficulty_path,
            self.selected_samples_path,
            self.deferred_queue_path,
            self.global_deferred_queue_path,
            self.discarded_samples_path,
            self.delta_cf_records_path,
            self.ratio_metrics_path,
            self.insights_path,
            self.insight_manifest_path,
            self.policy_context_path,
            self.context_prompt_path,
            self.context_injected_prompt_path,
            self.context_injected_agent_config_path,
            self.evaluation_metrics_path,
            self.ablation_manifest_path,
            self.evaluation_summary_path,
            self.metrics_path,
            self.config_path,
            self.manifest_path,
        )


def build_output_paths(
    *,
    output_root: Path,
    experiment_name: str,
    loop_index: int = 0,
) -> ContextEngineeringOutputPaths:
    experiment_root = output_root / experiment_name
    loop_dir = experiment_root / f"loop_{loop_index:03d}"
    return ContextEngineeringOutputPaths(
        experiment_root=experiment_root,
        loop_dir=loop_dir,
        trajectories_dir=loop_dir / "trajectories",
        counterfactual_dir=loop_dir / "counterfactual",
        difficulty_manifest_path=loop_dir / "difficulty_manifest.json",
        task_id_to_success_rate_path=loop_dir / "task_id_to_success_rate.json",
        task_id_to_difficulty_path=loop_dir / "task_id_to_difficulty.json",
        selected_samples_path=loop_dir / "selected_samples.jsonl",
        deferred_queue_path=loop_dir / "deferred_queue.jsonl",
        global_deferred_queue_path=experiment_root / "deferred_queue.jsonl",
        discarded_samples_path=loop_dir / "discarded_samples.jsonl",
        delta_cf_records_path=loop_dir / "delta_cf_records.jsonl",
        ratio_metrics_path=loop_dir / "ratio_metrics.json",
        insights_path=loop_dir / "insights.jsonl",
        insight_manifest_path=loop_dir / "insight_manifest.json",
        policy_context_path=experiment_root / "policy_context.jsonl",
        context_prompt_path=loop_dir / "context_prompt.md",
        context_injected_prompt_path=loop_dir / "context_injected_prompt.txt",
        context_injected_agent_config_path=loop_dir / "context_injected_agent_config.json",
        context_injected_runs_dir=loop_dir / "context_injected_runs",
        evaluation_metrics_path=loop_dir / "evaluation_metrics.json",
        ablation_manifest_path=loop_dir / "ablation_manifest.json",
        evaluation_summary_path=experiment_root / "evaluation_summary.json",
        metrics_path=experiment_root / "metrics.json",
        config_path=loop_dir / "context_engineering_config.json",
        manifest_path=loop_dir / "manifest.json",
    )


class ContextEngineeringRunner:
    def __init__(
        self,
        *,
        agent_config: dict[str, Any],
        dataset_name: str,
        config: ContextEngineeringConfig | dict[str, Any] | None = None,
        baseline_executor: BaselineTrajectoryExecutor | None = None,
        counterfactual_executor: CounterfactualReplayExecutor | None = None,
        ratio_sampler: RatioActionSampler | None = None,
        insight_extractor: InsightExtractor | None = None,
    ) -> None:
        self.agent_config = copy.deepcopy(agent_config)
        self.dataset_name = dataset_name
        self.config = load_context_engineering_config(config)
        self.baseline_executor = baseline_executor
        self.counterfactual_executor = counterfactual_executor
        self.ratio_sampler = ratio_sampler
        self.insight_extractor = insight_extractor

    def output_paths(self, experiment_name: str) -> ContextEngineeringOutputPaths:
        output_root = (
            Path(self.config.output.root) if self.config.output.root else _default_output_root()
        )
        return build_output_paths(
            output_root=output_root,
            experiment_name=experiment_name,
            loop_index=self.config.output.loop_index,
        )

    def initialize(self, experiment_name: str, task_ids: list[str] | None = None) -> dict[str, Any]:
        task_ids = task_ids or []
        paths = self.output_paths(experiment_name)
        paths.ensure_directories()
        _write_json(paths.config_path, self.config.to_dict())
        manifest = {
            "schema_version": 1,
            "phase": "phase_0_initialization",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "experiment_name": experiment_name,
            "dataset": self.dataset_name,
            "task_ids": task_ids,
            "num_task_ids": len(task_ids),
            "dry_run": self.config.dry_run,
            "output_paths": paths.to_dict(),
        }
        _write_json(paths.manifest_path, manifest)
        return manifest

    def solve_tasks(
        self,
        *,
        task_ids: list[str],
        experiment_name: str,
        num_processes: int = 1,
        process_index: int = 0,
    ) -> dict[str, Any] | None:
        assigned_task_ids = _assigned_task_ids(
            task_ids=task_ids, num_processes=num_processes, process_index=process_index
        )
        return self.solve_task_loop(
            task_ids=assigned_task_ids,
            experiment_name=experiment_name,
        )

    def solve_outer_loop(
        self,
        *,
        task_ids: list[str],
        experiment_name: str,
        num_loops: int | None = None,
        num_processes: int = 1,
        process_index: int = 0,
    ) -> dict[str, Any]:
        assigned_task_ids = _assigned_task_ids(
            task_ids=task_ids, num_processes=num_processes, process_index=process_index
        )
        max_loops = num_loops or self.config.outer_loop.max_loops
        if max_loops <= 0:
            raise ValueError(f"num_loops must be positive; got {max_loops}")
        loop_summaries: list[dict[str, Any]] = []
        start_loop_index = self.config.output.loop_index
        for loop_offset in range(max_loops):
            loop_index = start_loop_index + loop_offset
            loop_runner = self._runner_for_loop_index(loop_index)
            loop_manifest = loop_runner.solve_task_loop(
                task_ids=assigned_task_ids,
                experiment_name=experiment_name,
            )
            loop_summary = self._loop_summary_from_manifest(
                loop_manifest=loop_manifest,
                experiment_name=experiment_name,
                loop_index=loop_index,
            )
            loop_summaries.append(loop_summary)
            if self.config.outer_loop.stop_when_no_deferred and loop_summary["deferred_count"] == 0:
                break
        paths = self.output_paths(experiment_name)
        loop_evaluations = [
            _read_json_if_exists(
                build_output_paths(
                    output_root=paths.experiment_root.parent,
                    experiment_name=experiment_name,
                    loop_index=cast(int, loop_summary["loop_index"]),
                ).evaluation_metrics_path
            )
            for loop_summary in loop_summaries
        ]
        loop_evaluations = [
            loop_evaluation for loop_evaluation in loop_evaluations if loop_evaluation
        ]
        evaluation_summary = build_evaluation_summary(
            experiment_name=experiment_name,
            dataset_name=self.dataset_name,
            loop_evaluations=loop_evaluations,
        )
        metrics = {
            "schema_version": 1,
            "phase": "phase_9_outer_loop_orchestration",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "experiment_name": experiment_name,
            "dataset": self.dataset_name,
            "start_loop_index": start_loop_index,
            "num_requested_loops": max_loops,
            "num_loops_completed": len(loop_summaries),
            "loop_indices": [loop_summary["loop_index"] for loop_summary in loop_summaries],
            "task_ids": assigned_task_ids,
            "num_tasks": len(assigned_task_ids),
            "stop_when_no_deferred": self.config.outer_loop.stop_when_no_deferred,
            "policy_context_path": str(paths.policy_context_path),
            "global_deferred_queue_path": str(paths.global_deferred_queue_path),
            "evaluation_summary_path": str(paths.evaluation_summary_path),
            "loop_summaries": loop_summaries,
            "phase_10_evaluation_summary": evaluation_summary,
        }
        _write_json(paths.metrics_path, metrics)
        _write_json(paths.evaluation_summary_path, evaluation_summary)
        return metrics

    def solve_task_loop(
        self,
        *,
        task_ids: list[str],
        experiment_name: str,
    ) -> dict[str, Any]:
        self.initialize(experiment_name=experiment_name, task_ids=task_ids)
        if self.config.dry_run:
            return _read_json(self.output_paths(experiment_name).manifest_path)
        baseline_manifest = self.collect_baseline_trajectories(
            task_ids=task_ids,
            experiment_name=experiment_name,
        )
        curriculum_task_ids = _curriculum_task_ids_from_difficulty_manifest(
            baseline_manifest=baseline_manifest,
            difficulty_config=self.config.difficulty,
        )
        self.collect_counterfactual_pairs(
            task_ids=curriculum_task_ids,
            experiment_name=experiment_name,
        )
        self.extract_counterfactual_deltas(
            task_ids=curriculum_task_ids,
            experiment_name=experiment_name,
        )
        self.estimate_ratios(
            task_ids=curriculum_task_ids,
            experiment_name=experiment_name,
        )
        self.filter_ratio_samples(
            task_ids=curriculum_task_ids,
            experiment_name=experiment_name,
        )
        self.extract_insights(
            task_ids=curriculum_task_ids,
            experiment_name=experiment_name,
        )
        manifest = self.update_policy_context_and_run_context_injected_tasks(
            task_ids=curriculum_task_ids,
            experiment_name=experiment_name,
        )
        manifest["phase_9_curriculum"] = {
            "input_task_ids": task_ids,
            "curriculum_task_ids": curriculum_task_ids,
            "num_curriculum_task_ids": len(curriculum_task_ids),
        }
        _write_json(self.output_paths(experiment_name).manifest_path, manifest)
        return self.evaluate_phase10(
            task_ids=curriculum_task_ids,
            experiment_name=experiment_name,
        )

    def collect_baseline_trajectories(
        self, *, task_ids: list[str], experiment_name: str
    ) -> dict[str, Any]:
        paths = self.output_paths(experiment_name)
        paths.ensure_directories()
        baseline_executor = self.baseline_executor or AppWorldBaselineTrajectoryExecutor(
            self._sampling_agent_config()
        )
        task_summaries: list[dict[str, Any]] = []
        task_id_to_success_rate: dict[str, float] = {}
        task_id_to_difficulty: dict[str, float] = {}

        for task_id in task_ids:
            trajectory_records: list[BaselineTrajectoryRecord] = []
            for sample_index in range(self.config.sampling.n_difficulty):
                trajectory_id = _baseline_trajectory_id(task_id=task_id, sample_index=sample_index)
                trajectory_record = baseline_executor.run_trajectory(
                    task_id=task_id,
                    trajectory_id=trajectory_id,
                    loop_index=self.config.output.loop_index,
                    sample_index=sample_index,
                    random_seed=self._trajectory_seed(task_id=task_id, sample_index=sample_index),
                )
                trajectory_records.append(trajectory_record)
                _write_json(
                    paths.trajectories_dir / f"{trajectory_id}.json",
                    trajectory_record.to_dict(),
                )

            success_count = sum(
                1 for trajectory_record in trajectory_records if trajectory_record.success
            )
            success_rate = success_count / self.config.sampling.n_difficulty
            difficulty = 1.0 - success_rate
            task_id_to_success_rate[task_id] = success_rate
            task_id_to_difficulty[task_id] = difficulty
            task_summaries.append(
                {
                    "task_id": task_id,
                    "num_trajectories": len(trajectory_records),
                    "success_count": success_count,
                    "success_rate": success_rate,
                    "difficulty": difficulty,
                    "trajectory_ids": [
                        trajectory_record.trajectory_id for trajectory_record in trajectory_records
                    ],
                    "trajectory_paths": [
                        str(paths.trajectories_dir / f"{trajectory_record.trajectory_id}.json")
                        for trajectory_record in trajectory_records
                    ],
                }
            )

        difficulty_sorted_task_ids = [
            task_summary["task_id"]
            for task_summary in sorted(
                task_summaries,
                key=lambda task_summary: (
                    task_summary["difficulty"],
                    task_summary["task_id"],
                ),
            )
        ]
        manifest = {
            "schema_version": 1,
            "phase": "phase_1_baseline_trajectory_collection",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "experiment_name": experiment_name,
            "dataset": self.dataset_name,
            "loop_index": self.config.output.loop_index,
            "n_difficulty": self.config.sampling.n_difficulty,
            "num_tasks": len(task_ids),
            "task_ids": task_ids,
            "difficulty_sorted_task_ids": difficulty_sorted_task_ids,
            "task_summaries": task_summaries,
        }
        _write_json(paths.difficulty_manifest_path, manifest)
        _write_json(paths.task_id_to_success_rate_path, _sorted_mapping(task_id_to_success_rate))
        _write_json(paths.task_id_to_difficulty_path, _sorted_mapping(task_id_to_difficulty))
        _write_json(paths.manifest_path, manifest)
        return manifest

    def collect_counterfactual_pairs(
        self, *, task_ids: list[str], experiment_name: str
    ) -> dict[str, Any]:
        paths = self.output_paths(experiment_name)
        paths.ensure_directories()
        counterfactual_executor = (
            self.counterfactual_executor
            or ExistingCounterfactualReplayExecutor(
                agent_config=self._sampling_agent_config(),
                dataset_name=self.dataset_name,
                counterfactual_config=self.config.counterfactual.__dict__,
            )
        )
        pair_records: list[CounterfactualPairRecord] = []
        for task_id in task_ids:
            pair_record = counterfactual_executor.run_counterfactual_pair(
                task_id=task_id,
                experiment_name=f"{experiment_name}__counterfactual",
            )
            pair_records.append(pair_record)
            self._write_counterfactual_pair(
                paths=paths,
                pair_record=pair_record,
            )

        manifest = {
            "schema_version": 1,
            "phase": "phase_3_counterfactual_replay",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "experiment_name": experiment_name,
            "dataset": self.dataset_name,
            "loop_index": self.config.output.loop_index,
            "num_tasks": len(task_ids),
            "task_ids": task_ids,
            "pair_complete": all(pair_record.pair_complete for pair_record in pair_records),
            "task_manifests": {
                pair_record.task_id: pair_record.manifest() for pair_record in pair_records
            },
        }
        _write_json(paths.counterfactual_dir / "manifest.json", manifest)
        _write_json(paths.manifest_path, manifest)
        return manifest

    def extract_counterfactual_deltas(
        self, *, task_ids: list[str], experiment_name: str
    ) -> dict[str, Any]:
        paths = self.output_paths(experiment_name)
        delta_records: list[DeltaCfRecord] = []
        task_id_to_delta_records: dict[str, list[DeltaCfRecord]] = {}
        for task_id in task_ids:
            pair_record = self._read_counterfactual_pair(paths=paths, task_id=task_id)
            task_delta_records = extract_delta_cf_records(pair_record)
            task_id_to_delta_records[task_id] = task_delta_records
            delta_records.extend(task_delta_records)
            task_counterfactual_dir = paths.counterfactual_dir / _safe_path_component(task_id)
            _write_jsonl(
                task_counterfactual_dir / "delta_cf_records.jsonl",
                [record.to_dict() for record in task_delta_records],
            )

        manifest = {
            "schema_version": 1,
            "phase": "phase_4_delta_cf_extraction",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "experiment_name": experiment_name,
            "dataset": self.dataset_name,
            "loop_index": self.config.output.loop_index,
            "num_tasks": len(task_ids),
            "task_ids": task_ids,
            "num_delta_records": len(delta_records),
            "num_extraction_errors": sum(
                1 for record in delta_records if record.extraction_error is not None
            ),
            "task_delta_counts": {
                task_id: len(records)
                for task_id, records in sorted(task_id_to_delta_records.items())
            },
        }
        _write_jsonl(
            paths.delta_cf_records_path,
            [record.to_dict() for record in delta_records],
        )
        _write_json(paths.counterfactual_dir / "delta_cf_manifest.json", manifest)
        _write_json(paths.manifest_path, manifest)
        return manifest

    def estimate_ratios(self, *, task_ids: list[str], experiment_name: str) -> dict[str, Any]:
        paths = self.output_paths(experiment_name)
        paths.ensure_directories()
        ratio_sampler = self.ratio_sampler or AppWorldRatioActionSampler(
            self._sampling_agent_config()
        )
        ratio_records: list[RatioEstimateRecord] = []
        task_id_to_ratio_records: dict[str, list[RatioEstimateRecord]] = {}

        for task_id in task_ids:
            task_counterfactual_dir = paths.counterfactual_dir / _safe_path_component(task_id)
            task_delta_payloads = _read_jsonl(task_counterfactual_dir / "delta_cf_records.jsonl")
            task_ratio_records: list[RatioEstimateRecord] = []
            for delta_index, delta_payload in enumerate(task_delta_payloads):
                delta_record = _delta_record_from_mapping(delta_payload)
                raw_samples_path = (
                    task_counterfactual_dir / "ratio_raw_samples" / f"delta_{delta_index:03d}.jsonl"
                )
                old_samples, theta_samples = self._sample_ratio_actions(
                    ratio_sampler=ratio_sampler,
                    delta_record=delta_record,
                    delta_index=delta_index,
                )
                _write_jsonl(
                    raw_samples_path,
                    [sample.to_dict() for sample in [*old_samples, *theta_samples]],
                )
                ratio_record = estimate_ratio_from_samples(
                    delta_record=delta_record,
                    old_samples=old_samples,
                    theta_samples=theta_samples,
                    smoothing=self.config.smoothing,
                    raw_samples_path=str(raw_samples_path),
                )
                task_ratio_records.append(ratio_record)
                ratio_records.append(ratio_record)
            task_id_to_ratio_records[task_id] = task_ratio_records
            _write_jsonl(
                task_counterfactual_dir / "ratio_estimates.jsonl",
                [record.to_dict() for record in task_ratio_records],
            )

        ratio_payloads = [record.to_dict() for record in ratio_records]
        _write_jsonl(paths.loop_dir / "ratio_estimates.jsonl", ratio_payloads)
        metrics = {
            "schema_version": 1,
            "phase": "phase_5_ratio_estimation",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "experiment_name": experiment_name,
            "dataset": self.dataset_name,
            "loop_index": self.config.output.loop_index,
            "num_tasks": len(task_ids),
            "task_ids": task_ids,
            "num_ratio_records": len(ratio_records),
            "num_estimation_errors": sum(
                1 for ratio_record in ratio_records if ratio_record.estimation_error is not None
            ),
            "n_pi_old": self.config.sampling.n_pi_old,
            "n_pi_theta": self.config.sampling.n_pi_theta,
            "smoothing": {
                "K": self.config.smoothing.k,
                "alpha": self.config.smoothing.alpha,
                "action_match_level": self.config.smoothing.action_match_level,
            },
            "task_ratio_counts": {
                task_id: len(records)
                for task_id, records in sorted(task_id_to_ratio_records.items())
            },
            "ratio_records": ratio_payloads,
        }
        _write_json(paths.ratio_metrics_path, metrics)
        _write_json(paths.manifest_path, metrics)
        return metrics

    def filter_ratio_samples(self, *, task_ids: list[str], experiment_name: str) -> dict[str, Any]:
        paths = self.output_paths(experiment_name)
        paths.ensure_directories()
        ratio_payloads = _read_jsonl(paths.loop_dir / "ratio_estimates.jsonl")
        incoming_deferred_records = _incoming_deferred_records_for_loop(
            deferred_records=_read_jsonl_if_exists(paths.global_deferred_queue_path),
            loop_index=self.config.output.loop_index,
        )
        future_deferred_records = _future_deferred_records(
            deferred_records=_read_jsonl_if_exists(paths.global_deferred_queue_path),
            loop_index=self.config.output.loop_index,
        )
        ratio_payloads = _merge_ratio_payloads_with_deferred_records(
            ratio_payloads=ratio_payloads,
            deferred_records=incoming_deferred_records,
        )
        bucket_name_to_records = filter_ratio_records(
            ratio_records=ratio_payloads,
            ratio_filter_config=self.config.ratio_filter,
            loop_index=self.config.output.loop_index,
        )
        selected_records = bucket_name_to_records["selected"]
        deferred_records = bucket_name_to_records["deferred"]
        discarded_records = bucket_name_to_records["discarded"]

        _write_jsonl(
            paths.selected_samples_path,
            [record.to_dict() for record in selected_records],
        )
        _write_jsonl(
            paths.deferred_queue_path,
            [record.to_dict() for record in deferred_records],
        )
        _write_jsonl(
            paths.discarded_samples_path,
            [record.to_dict() for record in discarded_records],
        )
        _write_jsonl(
            paths.global_deferred_queue_path,
            [
                *future_deferred_records,
                *[record.to_dict() for record in deferred_records],
            ],
        )

        task_id_to_bucket_counts = _task_id_to_bucket_counts(
            [*selected_records, *deferred_records, *discarded_records]
        )
        manifest = {
            "schema_version": 1,
            "phase": "phase_6_ratio_filtering",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "experiment_name": experiment_name,
            "dataset": self.dataset_name,
            "loop_index": self.config.output.loop_index,
            "num_tasks": len(task_ids),
            "task_ids": task_ids,
            "epsilon": self.config.ratio_filter.epsilon,
            "lower_bound": 1.0 - self.config.ratio_filter.epsilon,
            "upper_bound": 1.0 + self.config.ratio_filter.epsilon,
            "high_ratio_policy": self.config.ratio_filter.high_ratio_policy,
            "low_ratio_policy": self.config.ratio_filter.low_ratio_policy,
            "max_defer_rounds": self.config.ratio_filter.max_defer_rounds,
            "num_ratio_records": len(ratio_payloads),
            "incoming_deferred_count": len(incoming_deferred_records),
            "future_deferred_count": len(future_deferred_records),
            "selected_count": len(selected_records),
            "deferred_count": len(deferred_records),
            "discarded_count": len(discarded_records),
            "all_records_bucketed_once": len(ratio_payloads)
            == len(selected_records) + len(deferred_records) + len(discarded_records),
            "task_bucket_counts": task_id_to_bucket_counts,
            "output_paths": {
                "selected_samples": str(paths.selected_samples_path),
                "deferred_queue": str(paths.deferred_queue_path),
                "global_deferred_queue": str(paths.global_deferred_queue_path),
                "discarded_samples": str(paths.discarded_samples_path),
                "ratio_metrics": str(paths.ratio_metrics_path),
            },
        }
        ratio_metrics = _read_json(paths.ratio_metrics_path)
        ratio_metrics["phase_6_filtering"] = manifest
        _write_json(paths.ratio_metrics_path, ratio_metrics)
        _write_json(paths.manifest_path, manifest)
        return manifest

    def extract_insights(self, *, task_ids: list[str], experiment_name: str) -> dict[str, Any]:
        paths = self.output_paths(experiment_name)
        paths.ensure_directories()
        selected_samples = _read_jsonl(paths.selected_samples_path)
        insight_records = extract_insight_records(
            selected_samples=selected_samples,
            insight_extractor=self.insight_extractor or HeuristicInsightExtractor(),
        )
        _write_jsonl(
            paths.insights_path,
            [insight_record.to_dict() for insight_record in insight_records],
        )
        task_id_to_insight_counts = _task_id_to_insight_counts(insight_records)
        manifest = {
            "schema_version": 1,
            "phase": "phase_7_insight_extraction",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "experiment_name": experiment_name,
            "dataset": self.dataset_name,
            "loop_index": self.config.output.loop_index,
            "num_tasks": len(task_ids),
            "task_ids": task_ids,
            "num_selected_samples": len(selected_samples),
            "num_insights": len(insight_records),
            "num_rejected_insights": sum(
                1 for insight_record in insight_records if insight_record.rejected
            ),
            "task_insight_counts": task_id_to_insight_counts,
            "output_paths": {
                "selected_samples": str(paths.selected_samples_path),
                "insights": str(paths.insights_path),
                "insight_manifest": str(paths.insight_manifest_path),
            },
        }
        _write_json(paths.insight_manifest_path, manifest)
        _write_json(paths.manifest_path, manifest)
        return manifest

    def update_policy_context_and_run_context_injected_tasks(
        self, *, task_ids: list[str], experiment_name: str
    ) -> dict[str, Any]:
        paths = self.output_paths(experiment_name)
        paths.ensure_directories()
        created_at_utc = datetime.now(UTC).isoformat()
        insight_payloads = _read_jsonl_if_exists(paths.insights_path)
        insight_records = [
            _insight_record_from_mapping(insight_payload) for insight_payload in insight_payloads
        ]
        new_context_items = build_policy_context_items(
            insight_records=insight_records,
            created_loop_index=self.config.output.loop_index,
            created_at_utc=created_at_utc,
            include_rejected=False,
        )
        existing_context_items = [
            PolicyContextItem.from_mapping(context_payload)
            for context_payload in _read_jsonl_if_exists(paths.policy_context_path)
        ]
        candidate_context_items = [*existing_context_items, *new_context_items]
        num_unique_candidate_context_items = len(
            {context_item.context_id for context_item in candidate_context_items}
        )
        policy_context_items = merge_policy_context_items(
            existing_items=existing_context_items,
            new_items=new_context_items,
            max_context_items=self.config.policy_context.max_context_items,
        )
        _write_jsonl(
            paths.policy_context_path,
            [context_item.to_dict() for context_item in policy_context_items],
        )
        context_prompt = render_policy_context_prompt(
            policy_context_items=policy_context_items,
            all_insight=self.config.policy_context.all_insight,
            max_context_items=self.config.policy_context.max_context_items,
        )
        _write_text(paths.context_prompt_path, context_prompt)
        injected_agent_config, prompt_injection_status = self._write_context_injected_agent_config(
            paths=paths
        )
        context_run_summaries = self._run_context_injected_trajectories(
            paths=paths,
            task_ids=task_ids,
            injected_agent_config=injected_agent_config,
            enabled=bool(policy_context_items),
        )
        manifest = {
            "schema_version": 1,
            "phase": "phase_8_policy_context_prompt_injection",
            "created_at_utc": created_at_utc,
            "experiment_name": experiment_name,
            "dataset": self.dataset_name,
            "loop_index": self.config.output.loop_index,
            "num_tasks": len(task_ids),
            "task_ids": task_ids,
            "num_insights_read": len(insight_records),
            "num_rejected_insights_read": sum(
                1 for insight_record in insight_records if insight_record.rejected
            ),
            "num_new_context_items": len(new_context_items),
            "num_policy_context_items": len(policy_context_items),
            "num_deduplicated_items": len(candidate_context_items)
            - num_unique_candidate_context_items,
            "num_trimmed_context_items": num_unique_candidate_context_items
            - len(policy_context_items),
            "all_insight": self.config.policy_context.all_insight,
            "max_context_items": self.config.policy_context.max_context_items,
            "context_prompt_token_count": estimate_token_count(context_prompt),
            "prompt_injection": prompt_injection_status,
            "context_run_summaries": context_run_summaries,
            "output_paths": {
                "policy_context": str(paths.policy_context_path),
                "context_prompt": str(paths.context_prompt_path),
                "context_injected_prompt": str(paths.context_injected_prompt_path),
                "context_injected_agent_config": str(paths.context_injected_agent_config_path),
                "context_injected_runs": str(paths.context_injected_runs_dir),
            },
        }
        _write_json(paths.context_injected_runs_dir / "manifest.json", manifest)
        _write_json(paths.manifest_path, manifest)
        return manifest

    def evaluate_phase10(self, *, task_ids: list[str], experiment_name: str) -> dict[str, Any]:
        paths = self.output_paths(experiment_name)
        paths.ensure_directories()
        evaluation_metrics = build_phase10_evaluation(
            experiment_name=experiment_name,
            dataset_name=self.dataset_name,
            loop_index=self.config.output.loop_index,
            task_ids=task_ids,
            trajectories_dir=paths.trajectories_dir,
            context_injected_runs_dir=paths.context_injected_runs_dir,
            policy_context_path=paths.policy_context_path,
            ratio_estimates_path=paths.loop_dir / "ratio_estimates.jsonl",
            ratio_metrics_path=paths.ratio_metrics_path,
            deferred_queue_path=paths.global_deferred_queue_path,
            manifest_path=paths.manifest_path,
            ablation_settings=self.config.evaluation.ablation_settings,
        )
        previous_manifest = _read_json_if_exists(paths.manifest_path)
        manifest = {
            **previous_manifest,
            "phase": "phase_10_evaluation_ablation",
            "phase_10_evaluation": evaluation_metrics,
            "output_paths": {
                **cast(dict[str, Any], previous_manifest.get("output_paths", {})),
                "evaluation_metrics": str(paths.evaluation_metrics_path),
                "ablation_manifest": str(paths.ablation_manifest_path),
            },
        }
        _write_json(paths.evaluation_metrics_path, evaluation_metrics)
        _write_json(
            paths.ablation_manifest_path,
            {
                "schema_version": 1,
                "phase": "phase_10_ablation_manifest",
                "experiment_name": experiment_name,
                "dataset": self.dataset_name,
                "loop_index": self.config.output.loop_index,
                "ablation_settings": evaluation_metrics["ablation_settings"],
                "ablation_results": evaluation_metrics["ablation_results"],
            },
        )
        _write_json(paths.manifest_path, manifest)
        return manifest

    def _sample_ratio_actions(
        self,
        *,
        ratio_sampler: RatioActionSampler,
        delta_record: DeltaCfRecord,
        delta_index: int,
    ) -> tuple[list[ActionSampleRecord], list[ActionSampleRecord]]:
        if delta_record.state_snapshot is None:
            return [], []
        messages_before_action = (
            delta_record.messages_before_action
            if delta_record.messages_before_action is not None
            else messages_from_state_snapshot(delta_record.state_snapshot)
        )
        success_action_signature_payload = delta_record.delta_cf.get("success_action_signature")
        success_action_signature = None
        if isinstance(success_action_signature_payload, dict):
            from appworld_agents.code.context_engineering.action import ActionSignature

            success_action_signature = ActionSignature(
                app_name=str(success_action_signature_payload["app_name"]),
                api_name=str(success_action_signature_payload["api_name"]),
                argument_keys=tuple(
                    str(key) for key in success_action_signature_payload.get("argument_keys", [])
                ),
            )
        old_samples = [
            ratio_sampler.sample_action(
                task_id=delta_record.task_id,
                timestep=delta_record.timestep,
                state_snapshot=delta_record.state_snapshot,
                messages_before_action=messages_before_action,
                policy_name="pi_old",
                sample_index=sample_index,
                random_seed=self._ratio_seed(
                    task_id=delta_record.task_id,
                    delta_index=delta_index,
                    policy_offset=0,
                    sample_index=sample_index,
                ),
                success_action_signature=success_action_signature,
                action_match_level=self.config.smoothing.action_match_level,
            )
            for sample_index in range(self.config.sampling.n_pi_old)
        ]
        theta_samples = [
            ratio_sampler.sample_action(
                task_id=delta_record.task_id,
                timestep=delta_record.timestep,
                state_snapshot=delta_record.state_snapshot,
                messages_before_action=messages_before_action,
                policy_name="pi_theta",
                sample_index=sample_index,
                random_seed=self._ratio_seed(
                    task_id=delta_record.task_id,
                    delta_index=delta_index,
                    policy_offset=10_000,
                    sample_index=sample_index,
                ),
                success_action_signature=success_action_signature,
                action_match_level=self.config.smoothing.action_match_level,
            )
            for sample_index in range(self.config.sampling.n_pi_theta)
        ]
        return old_samples, theta_samples

    def _sampling_agent_config(self) -> dict[str, Any]:
        agent_config = copy.deepcopy(self.agent_config)
        model_config = dict(agent_config.get("model_config", {}) or {})
        model_config["temperature"] = self.config.sampling.temperature_sampling
        agent_config["model_config"] = model_config
        agent_config.setdefault("max_steps", self.config.appworld.max_steps)
        return agent_config

    def _write_context_injected_agent_config(
        self, *, paths: ContextEngineeringOutputPaths
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        agent_config = self._sampling_agent_config()
        context_prompt = _read_text(paths.context_prompt_path)
        prompt_file_path = agent_config.get("prompt_file_path")
        if isinstance(prompt_file_path, str) and Path(prompt_file_path).exists():
            original_prompt_template = _read_text(Path(prompt_file_path))
            injected_prompt_template = inject_policy_context_prompt_template(
                original_prompt_template=original_prompt_template,
                context_prompt=context_prompt,
            )
            _write_text(paths.context_injected_prompt_path, injected_prompt_template)
            agent_config["prompt_file_path"] = str(paths.context_injected_prompt_path)
            status = {
                "mode": "prompt_file_path",
                "original_prompt_file_path": prompt_file_path,
                "context_injected_prompt_file_path": str(paths.context_injected_prompt_path),
                "context_prompt_file_path": str(paths.context_prompt_path),
            }
        else:
            agent_config.setdefault("context_engineering", {})
            agent_config["context_engineering"]["context_prompt_file_path"] = str(
                paths.context_prompt_path
            )
            status = {
                "mode": "metadata_only",
                "reason": "agent_config.prompt_file_path_missing_or_unreadable",
                "context_prompt_file_path": str(paths.context_prompt_path),
            }
        _write_json(paths.context_injected_agent_config_path, agent_config)
        return agent_config, status

    def _run_context_injected_trajectories(
        self,
        *,
        paths: ContextEngineeringOutputPaths,
        task_ids: list[str],
        injected_agent_config: dict[str, Any],
        enabled: bool,
    ) -> list[dict[str, Any]]:
        if not enabled:
            return [
                {
                    "skipped": True,
                    "reason": "no_policy_context_items",
                    "num_task_ids": len(task_ids),
                }
            ]
        context_executor = self.baseline_executor or AppWorldBaselineTrajectoryExecutor(
            injected_agent_config
        )
        run_summaries: list[dict[str, Any]] = []
        for task_id in task_ids:
            trajectory_id = (
                f"{_safe_path_component(task_id)}__context_injected_"
                f"{self.config.output.loop_index:03d}"
            )
            random_seed = self._context_injected_seed(task_id=task_id)
            trajectory_record = context_executor.run_trajectory(
                task_id=task_id,
                trajectory_id=trajectory_id,
                loop_index=self.config.output.loop_index,
                sample_index=0,
                random_seed=random_seed,
            )
            trajectory_path = paths.context_injected_runs_dir / f"{trajectory_id}.json"
            _write_json(trajectory_path, trajectory_record.to_dict())
            run_summaries.append(
                {
                    "task_id": task_id,
                    "trajectory_id": trajectory_id,
                    "trajectory_path": str(trajectory_path),
                    "random_seed": random_seed,
                    "success": trajectory_record.success,
                    "error": trajectory_record.error,
                }
            )
        return run_summaries

    def _trajectory_seed(self, *, task_id: str, sample_index: int) -> int | None:
        base_seed = self._base_random_seed()
        if base_seed is None:
            return None
        return base_seed + _stable_task_offset(task_id) + sample_index

    def _ratio_seed(
        self,
        *,
        task_id: str,
        delta_index: int,
        policy_offset: int,
        sample_index: int,
    ) -> int | None:
        base_seed = self._base_random_seed()
        if base_seed is None:
            return None
        return (
            base_seed
            + _stable_task_offset(task_id)
            + policy_offset
            + delta_index * 1_000
            + sample_index
        )

    def _context_injected_seed(self, *, task_id: str) -> int | None:
        base_seed = self._base_random_seed()
        if base_seed is None:
            return None
        return base_seed + _stable_task_offset(task_id) + 100_000

    def _runner_for_loop_index(self, loop_index: int) -> ContextEngineeringRunner:
        config = replace(
            self.config,
            output=replace(self.config.output, loop_index=loop_index),
        )
        return ContextEngineeringRunner(
            agent_config=self.agent_config,
            dataset_name=self.dataset_name,
            config=config,
            baseline_executor=self.baseline_executor,
            counterfactual_executor=self.counterfactual_executor,
            ratio_sampler=self.ratio_sampler,
            insight_extractor=self.insight_extractor,
        )

    def _loop_summary_from_manifest(
        self,
        *,
        loop_manifest: dict[str, Any] | None,
        experiment_name: str,
        loop_index: int,
    ) -> dict[str, Any]:
        paths = build_output_paths(
            output_root=(
                Path(self.config.output.root) if self.config.output.root else _default_output_root()
            ),
            experiment_name=experiment_name,
            loop_index=loop_index,
        )
        filter_manifest = _read_json_if_exists(paths.ratio_metrics_path).get(
            "phase_6_filtering", {}
        )
        evaluation_metrics = _read_json_if_exists(paths.evaluation_metrics_path)
        policy_context_count = len(_read_jsonl_if_exists(paths.policy_context_path))
        curriculum = {}
        if loop_manifest:
            curriculum = cast(dict[str, Any], loop_manifest.get("phase_9_curriculum", {}))
        return {
            "loop_index": loop_index,
            "loop_dir": str(paths.loop_dir),
            "manifest_path": str(paths.manifest_path),
            "input_task_ids": curriculum.get("input_task_ids", []),
            "curriculum_task_ids": curriculum.get("curriculum_task_ids", []),
            "selected_count": int(filter_manifest.get("selected_count", 0)),
            "deferred_count": int(filter_manifest.get("deferred_count", 0)),
            "discarded_count": int(filter_manifest.get("discarded_count", 0)),
            "incoming_deferred_count": int(filter_manifest.get("incoming_deferred_count", 0)),
            "num_policy_context_items": policy_context_count,
            "success_rate": evaluation_metrics.get("success_rate"),
            "delta_success_rate": evaluation_metrics.get("delta_success_rate"),
            "evaluation_metrics_path": str(paths.evaluation_metrics_path),
            "phase": loop_manifest.get("phase") if loop_manifest else None,
        }

    def _base_random_seed(self) -> int | None:
        appworld_config = self.agent_config.get("appworld_config", {}) or {}
        raw_seed = appworld_config.get("random_seed")
        return int(raw_seed) if raw_seed is not None else None

    def _write_counterfactual_pair(
        self,
        *,
        paths: ContextEngineeringOutputPaths,
        pair_record: CounterfactualPairRecord,
    ) -> None:
        task_counterfactual_dir = paths.counterfactual_dir / _safe_path_component(
            pair_record.task_id
        )
        trajectories_dir = task_counterfactual_dir / "trajectories"
        trajectories_dir.mkdir(parents=True, exist_ok=True)
        _write_json(task_counterfactual_dir / "manifest.json", pair_record.manifest())
        for trajectory in pair_record.trajectories:
            _write_json(
                trajectories_dir / f"{_safe_path_component(trajectory.trajectory_id)}.json",
                trajectory.to_dict(),
            )

    def _read_counterfactual_pair(
        self,
        *,
        paths: ContextEngineeringOutputPaths,
        task_id: str,
    ) -> CounterfactualPairRecord:
        task_counterfactual_dir = paths.counterfactual_dir / _safe_path_component(task_id)
        manifest = _read_json(task_counterfactual_dir / "manifest.json")
        trajectories = []
        for trajectory_id in manifest["trajectory_ids"]:
            trajectory_payload = _read_json(
                task_counterfactual_dir
                / "trajectories"
                / f"{_safe_path_component(str(trajectory_id))}.json"
            )
            trajectories.append(CounterfactualTrajectoryRecord.from_mapping(trajectory_payload))
        return CounterfactualPairRecord(
            task_id=task_id,
            case=cast(str, manifest["case"]),
            root_success=bool(manifest["root_success"]),
            final_success_trajectory_id=cast(str | None, manifest["final_success_trajectory_id"]),
            final_failure_trajectory_id=cast(str | None, manifest["final_failure_trajectory_id"]),
            pair_complete=bool(manifest["pair_complete"]),
            trajectories=trajectories,
            counterfactual_config=cast(dict[str, Any], manifest.get("counterfactual_config", {})),
        )


def extract_dataset_name(runner_config: dict[str, Any]) -> str:
    if "dataset" not in runner_config:
        raise Exception("Dataset name not found in the runner config.")
    return cast(str, runner_config["dataset"])


def run_experiment(
    experiment_name: str,
    runner_config: dict[str, Any],
    task_id: str | None = None,
    num_processes: int = 1,
    process_index: int = 0,
) -> None:
    runner_config = copy.deepcopy(runner_config)
    agent_config = runner_config.pop("agent", {})
    dataset_name = runner_config.pop("dataset")
    context_engineering_config = runner_config.pop("context_engineering", {})
    if runner_config:
        raise Exception(f"Unexpected keys in the runner config: {runner_config}")
    runner = ContextEngineeringRunner(
        agent_config=agent_config,
        dataset_name=dataset_name,
        config=context_engineering_config,
    )
    if task_id:
        task_ids = [task_id]
    else:
        from appworld.task import Task, load_task_ids

        task_ids = load_task_ids(dataset_name)
        for task_id_ in task_ids:
            Task.load(task_id=task_id_)
    if runner.config.outer_loop.max_loops > 1:
        runner.solve_outer_loop(
            task_ids=task_ids,
            experiment_name=experiment_name,
            num_processes=num_processes,
            process_index=process_index,
        )
    else:
        runner.solve_tasks(
            task_ids=task_ids,
            experiment_name=experiment_name,
            num_processes=num_processes,
            process_index=process_index,
        )


def _default_output_root() -> Path:
    from appworld.common.path_store import path_store

    return Path(path_store.experiment_outputs)


def _write_json(file_path: Path, payload: dict[str, Any]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def _read_json(file_path: Path) -> dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as file:
        return cast(dict[str, Any], json.load(file))


def _read_json_if_exists(file_path: Path) -> dict[str, Any]:
    if not file_path.exists():
        return {}
    return _read_json(file_path)


def _read_jsonl(file_path: Path) -> list[dict[str, Any]]:
    with file_path.open("r", encoding="utf-8") as file:
        return [cast(dict[str, Any], json.loads(line)) for line in file if line.strip()]


def _read_jsonl_if_exists(file_path: Path) -> list[dict[str, Any]]:
    if not file_path.exists():
        return []
    return _read_jsonl(file_path)


def _write_jsonl(file_path: Path, records: list[dict[str, Any]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True))
            file.write("\n")


def _write_text(file_path: Path, text: str) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(text, encoding="utf-8")


def _read_text(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


def _delta_record_from_mapping(data: dict[str, Any]) -> DeltaCfRecord:
    return DeltaCfRecord(
        task_id=cast(str, data["task_id"]),
        timestep=cast(int | None, data.get("timestep")),
        success_trajectory_id=cast(str | None, data.get("success_trajectory_id")),
        failure_trajectory_id=cast(str | None, data.get("failure_trajectory_id")),
        success_action=cast(str | None, data.get("success_action")),
        failure_action=cast(str | None, data.get("failure_action")),
        delta_cf=cast(dict[str, Any], data.get("delta_cf", {})),
        state_snapshot=cast(str | None, data.get("state_snapshot")),
        messages_before_action=cast(
            list[dict[str, Any]] | None, data.get("messages_before_action")
        ),
        state_source_trajectory_id=cast(str | None, data.get("state_source_trajectory_id")),
        extraction_error=cast(str | None, data.get("extraction_error")),
    )


def _insight_record_from_mapping(data: dict[str, Any]) -> InsightRecord:
    return InsightRecord(
        task_id=cast(str, data["task_id"]),
        timestep=cast(int | None, data.get("timestep")),
        state_summary=cast(str, data.get("state_summary", "")),
        success_action=cast(str | None, data.get("success_action")),
        failure_action=cast(str | None, data.get("failure_action")),
        delta_cf=cast(dict[str, Any], data.get("delta_cf", {})),
        insight=cast(str, data["insight"]),
        ratio=cast(float | None, data.get("ratio")),
        source=cast(str, data.get("source", "")),
        selected_sample_id=cast(str, data.get("selected_sample_id", "")),
        evidence=cast(dict[str, Any], data.get("evidence", {})),
        validation_flags=cast(list[str], data.get("validation_flags", [])),
        rejected=bool(data.get("rejected", False)),
    )


def _curriculum_task_ids_from_difficulty_manifest(
    *,
    baseline_manifest: dict[str, Any],
    difficulty_config: Any,
) -> list[str]:
    sorted_task_ids = [
        str(task_id) for task_id in baseline_manifest.get("difficulty_sorted_task_ids", [])
    ]
    task_id_to_difficulty = {
        str(task_summary["task_id"]): float(task_summary.get("difficulty", 0.0))
        for task_summary in baseline_manifest.get("task_summaries", [])
        if isinstance(task_summary, dict) and "task_id" in task_summary
    }
    filtered_task_ids = [
        task_id
        for task_id in sorted_task_ids
        if difficulty_config.d_min
        <= task_id_to_difficulty.get(task_id, 0.0)
        <= difficulty_config.d_max
    ]
    return filtered_task_ids or sorted_task_ids


def _incoming_deferred_records_for_loop(
    *, deferred_records: list[dict[str, Any]], loop_index: int
) -> list[dict[str, Any]]:
    return [
        deferred_record
        for deferred_record in deferred_records
        if _optional_int(deferred_record.get("next_loop_index")) == loop_index
    ]


def _future_deferred_records(
    *, deferred_records: list[dict[str, Any]], loop_index: int
) -> list[dict[str, Any]]:
    future_records: list[dict[str, Any]] = []
    for deferred_record in deferred_records:
        next_loop_index = _optional_int(deferred_record.get("next_loop_index"))
        if next_loop_index is not None and next_loop_index > loop_index:
            future_records.append(deferred_record)
    return future_records


def _merge_ratio_payloads_with_deferred_records(
    *,
    ratio_payloads: list[dict[str, Any]],
    deferred_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not deferred_records:
        return ratio_payloads
    merged_ratio_payloads = [copy.deepcopy(ratio_payload) for ratio_payload in ratio_payloads]
    ratio_key_to_index = {
        _ratio_record_key(ratio_payload): ratio_index
        for ratio_index, ratio_payload in enumerate(merged_ratio_payloads)
    }
    for deferred_record in deferred_records:
        carried_ratio_record = _carried_ratio_record_from_deferred_record(deferred_record)
        ratio_key = _ratio_record_key(carried_ratio_record)
        if ratio_key in ratio_key_to_index:
            existing_ratio_record = merged_ratio_payloads[ratio_key_to_index[ratio_key]]
            prior_defer_round = max(
                _optional_int(existing_ratio_record.get("prior_defer_round")) or 0,
                _optional_int(carried_ratio_record.get("prior_defer_round")) or 0,
            )
            existing_ratio_record["prior_defer_round"] = prior_defer_round
            existing_ratio_record.setdefault("deferred_from_sample_ids", []).extend(
                carried_ratio_record.get("deferred_from_sample_ids", [])
            )
            continue
        ratio_key_to_index[ratio_key] = len(merged_ratio_payloads)
        merged_ratio_payloads.append(carried_ratio_record)
    return merged_ratio_payloads


def _carried_ratio_record_from_deferred_record(
    deferred_record: dict[str, Any],
) -> dict[str, Any]:
    ratio_record = copy.deepcopy(deferred_record.get("ratio_record") or {})
    if not isinstance(ratio_record, dict):
        ratio_record = {}
    ratio_record["prior_defer_round"] = (
        _optional_int(deferred_record.get("defer_round"))
        or _optional_int(ratio_record.get("defer_round"))
        or 0
    )
    ratio_record["deferred_from_sample_ids"] = [str(deferred_record.get("sample_id", ""))]
    ratio_record["deferred_from_reason"] = deferred_record.get("reason")
    return ratio_record


def _ratio_record_key(ratio_record: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(ratio_record.get("task_id", "")),
        str(ratio_record.get("timestep", "")),
        str(ratio_record.get("success_trajectory_id", "")),
        str(ratio_record.get("failure_trajectory_id", "")),
        str(ratio_record.get("success_action", "")),
        str(ratio_record.get("failure_action", "")),
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _task_id_to_bucket_counts(
    records: list[RatioFilterDecisionRecord],
) -> dict[str, dict[str, int]]:
    task_id_to_counts: dict[str, dict[str, int]] = {}
    for record in records:
        bucket_to_count = task_id_to_counts.setdefault(
            record.task_id,
            {"selected": 0, "deferred": 0, "discarded": 0},
        )
        bucket_to_count[record.bucket] += 1
    return {task_id: task_id_to_counts[task_id] for task_id in sorted(task_id_to_counts)}


def _task_id_to_insight_counts(records: list[InsightRecord]) -> dict[str, int]:
    task_id_to_count: dict[str, int] = {}
    for record in records:
        task_id_to_count[record.task_id] = task_id_to_count.get(record.task_id, 0) + 1
    return {task_id: task_id_to_count[task_id] for task_id in sorted(task_id_to_count)}


def _assigned_task_ids(*, task_ids: list[str], num_processes: int, process_index: int) -> list[str]:
    if not task_ids:
        return []
    if num_processes <= 1:
        return task_ids
    from appworld.common.collections import chunk_and_return

    num_processes = min(num_processes, len(task_ids))
    return cast(
        list[str],
        chunk_and_return(
            task_ids, num_chunks=num_processes, chunk_index=process_index, balanced=True
        ),
    )


def _baseline_trajectory_id(*, task_id: str, sample_index: int) -> str:
    safe_task_id = _safe_path_component(task_id)
    return f"{safe_task_id}__baseline_{sample_index:03d}"


def _safe_path_component(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "_.-" else "_" for character in value
    )


def _stable_task_offset(task_id: str) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(task_id))


def _sorted_mapping(mapping: dict[str, float]) -> dict[str, float]:
    return {key: mapping[key] for key in sorted(mapping)}
