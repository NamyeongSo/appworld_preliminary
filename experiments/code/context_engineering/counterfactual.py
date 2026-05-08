from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from appworld_agents.code.context_engineering.action import build_step_representation


@dataclass(frozen=True)
class CounterfactualTrajectoryRecord:
    trajectory_id: str
    task_id: str
    case: str
    attempt_index: int
    parent_trajectory_id: str | None
    source_trajectory_id: str | None
    modified_timestep: int | None
    original_action: str | None
    modified_action: str | None
    modification_prompt: str | None
    action_changed: bool
    evaluation: dict[str, Any]
    steps: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    environment_io: list[dict[str, Any]]
    replay_objective_met: bool
    is_final_pair_member: bool = False
    error: str | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> CounterfactualTrajectoryRecord:
        return cls(
            trajectory_id=cast(str, data["trajectory_id"]),
            task_id=cast(str, data["task_id"]),
            case=cast(str, data["case"]),
            attempt_index=int(data["attempt_index"]),
            parent_trajectory_id=cast(str | None, data.get("parent_trajectory_id")),
            source_trajectory_id=cast(str | None, data.get("source_trajectory_id")),
            modified_timestep=cast(int | None, data.get("modified_timestep")),
            original_action=cast(str | None, data.get("original_action")),
            modified_action=cast(str | None, data.get("modified_action")),
            modification_prompt=cast(str | None, data.get("modification_prompt")),
            action_changed=bool(data.get("action_changed", False)),
            evaluation=cast(dict[str, Any], data.get("evaluation", {})),
            steps=cast(list[dict[str, Any]], data.get("steps", [])),
            messages=cast(list[dict[str, Any]], data.get("messages", [])),
            environment_io=cast(list[dict[str, Any]], data.get("environment_io", [])),
            replay_objective_met=bool(data.get("replay_objective_met", False)),
            is_final_pair_member=bool(data.get("is_final_pair_member", False)),
            error=cast(str | None, data.get("error")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "task_id": self.task_id,
            "case": self.case,
            "attempt_index": self.attempt_index,
            "parent_trajectory_id": self.parent_trajectory_id,
            "source_trajectory_id": self.source_trajectory_id,
            "modified_timestep": self.modified_timestep,
            "original_action": self.original_action,
            "modified_action": self.modified_action,
            "action_changed": self.action_changed,
            "modification_prompt": self.modification_prompt,
            "evaluation": self.evaluation,
            "replay_objective_met": self.replay_objective_met,
            "is_final_pair_member": self.is_final_pair_member,
            "error": self.error,
            "steps": [_enrich_step(step) for step in self.steps],
            "messages": self.messages,
            "environment_io": self.environment_io,
        }


@dataclass(frozen=True)
class CounterfactualPairRecord:
    task_id: str
    case: str
    root_success: bool
    final_success_trajectory_id: str | None
    final_failure_trajectory_id: str | None
    pair_complete: bool
    trajectories: list[CounterfactualTrajectoryRecord]
    counterfactual_config: dict[str, Any]

    @property
    def trajectory_ids(self) -> list[str]:
        return [trajectory.trajectory_id for trajectory in self.trajectories]

    def manifest(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "case": self.case,
            "root_success": self.root_success,
            "final_success_trajectory_id": self.final_success_trajectory_id,
            "final_failure_trajectory_id": self.final_failure_trajectory_id,
            "pair_complete": self.pair_complete,
            "trajectory_ids": self.trajectory_ids,
            "num_trajectories": len(self.trajectories),
            "counterfactual_config": self.counterfactual_config,
            "final_pair_actions": self._final_pair_actions(),
        }

    def _final_pair_actions(self) -> dict[str, Any]:
        final_pair_trajectories = {
            trajectory.trajectory_id: trajectory
            for trajectory in self.trajectories
            if trajectory.trajectory_id
            in {
                self.final_success_trajectory_id,
                self.final_failure_trajectory_id,
            }
        }
        return {
            trajectory_id: {
                "modified_timestep": trajectory.modified_timestep,
                "original_action": trajectory.original_action,
                "modified_action": trajectory.modified_action,
                "action_changed": trajectory.action_changed,
            }
            for trajectory_id, trajectory in sorted(final_pair_trajectories.items())
        }


class CounterfactualReplayExecutor(Protocol):
    def run_counterfactual_pair(
        self,
        *,
        task_id: str,
        experiment_name: str,
    ) -> CounterfactualPairRecord: ...


class ExistingCounterfactualReplayExecutor:
    """Adapter around the existing AppWorld counterfactual replay runner."""

    def __init__(
        self,
        *,
        agent_config: dict[str, Any],
        dataset_name: str,
        counterfactual_config: dict[str, Any] | None = None,
    ) -> None:
        self.agent_config = copy.deepcopy(agent_config)
        self.dataset_name = dataset_name
        self.counterfactual_config = counterfactual_config or {}

    def run_counterfactual_pair(
        self,
        *,
        task_id: str,
        experiment_name: str,
    ) -> CounterfactualPairRecord:
        from appworld.common.path_store import path_store
        from appworld_agents.code.counterfactual.run import CounterfactualReplayRunner

        runner = CounterfactualReplayRunner(
            agent_config=self.agent_config,
            dataset_name=self.dataset_name,
            counterfactual_config=self._existing_runner_config(),
        )
        runner._experiment_name = experiment_name
        runner.solve_task(task_id=task_id)

        task_counterfactual_dir = (
            Path(path_store.experiment_outputs)
            / experiment_name
            / "tasks"
            / task_id
            / "counterfactual"
        )
        manifest_path = task_counterfactual_dir / "manifest.json"
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)

        trajectories = []
        for trajectory_id in manifest["trajectory_ids"]:
            trajectory_path = task_counterfactual_dir / "trajectories" / f"{trajectory_id}.json"
            with trajectory_path.open("r", encoding="utf-8") as file:
                trajectories.append(CounterfactualTrajectoryRecord.from_mapping(json.load(file)))

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

    def _existing_runner_config(self) -> dict[str, Any]:
        config = copy.deepcopy(self.counterfactual_config)
        if "max_replay_attempts" not in config:
            max_backward_steps = config.get("max_backward_steps", "T")
            config["max_replay_attempts"] = (
                max_backward_steps if isinstance(max_backward_steps, int) else 50
            )
        if "case_b_max_attempts" not in config:
            config["case_b_max_attempts"] = config.get("max_success_to_failure_attempts", 5)
        config.setdefault("copy_attempt_dbs", False)
        return config


def _enrich_step(step: dict[str, Any]) -> dict[str, Any]:
    enriched_step = copy.deepcopy(step)
    if {
        "state_snapshot",
        "action_signature",
        "action_signatures",
        "action_parse_error",
    }.issubset(enriched_step):
        return enriched_step
    messages_before_action = cast(
        list[dict[str, Any]], enriched_step.get("messages_before_action", [])
    )
    action = cast(str, enriched_step.get("action", ""))
    enriched_step.update(
        build_step_representation(
            messages_before_action=messages_before_action,
            action=action,
        ).to_dict()
    )
    return enriched_step
