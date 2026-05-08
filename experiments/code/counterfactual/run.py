from __future__ import annotations

import copy
import json
import os
import random
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from appworld import AppWorld
from appworld.common.collections import chunk_and_return
from appworld.common.io import maybe_create_parent_directory, write_file, write_json
from appworld.common.path_store import path_store
from appworld.common.random import set_random_seed
from appworld.task import Task, load_task_ids
from appworld_agents.code.common.usage_tracker import Usage
from appworld_agents.code.simplified.agent import Agent, ExecutionIO, Status
from appworld_agents.code.simplified.react_code_agent import SimplifiedReActCodeAgent


@dataclass
class StepRecord:
    step_index: int
    step_number: int
    messages_before_action: list[dict[str, Any]]
    assistant_message: dict[str, Any]
    action: str
    environment_output: str
    usage: dict[str, Any]
    status: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryRecord:
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
    steps: list[StepRecord]
    messages: list[dict[str, Any]]
    environment_io: list[dict[str, Any]]
    replay_objective_met: bool
    is_final_pair_member: bool = False
    error: str | None = None

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
            "steps": [step.__dict__ for step in self.steps],
            "messages": self.messages,
            "environment_io": self.environment_io,
        }


def extract_dataset_name(runner_config: dict[str, Any]) -> str:
    if "dataset" not in runner_config:
        raise Exception("Dataset name not found in the runner config.")
    return cast(str, runner_config["dataset"])


class CounterfactualReplayRunner:
    def __init__(
        self,
        agent_config: dict[str, Any],
        dataset_name: str,
        counterfactual_config: dict[str, Any] | None = None,
    ) -> None:
        self.agent_config = copy.deepcopy(agent_config)
        self.dataset_name = dataset_name
        self.counterfactual_config = counterfactual_config or {}
        self.max_replay_attempts = int(self.counterfactual_config.get("max_replay_attempts", 50))
        self.case_b_max_attempts = int(self.counterfactual_config.get("case_b_max_attempts", 50))
        self.random_seed = self.counterfactual_config.get(
            "random_seed", self._agent_appworld_config.get("random_seed", None)
        )
        self.copy_attempt_dbs = bool(self.counterfactual_config.get("copy_attempt_dbs", True))
        self.ensure_different_action = bool(
            self.counterfactual_config.get("ensure_different_action", True)
        )
        self._experiment_name: str | None = None
        self._run_random = random.Random(self.random_seed)

    @property
    def _agent_appworld_config(self) -> dict[str, Any]:
        return cast(dict[str, Any], self.agent_config.get("appworld_config", {}) or {})

    def _make_agent(self) -> SimplifiedReActCodeAgent:
        # Keep the original agent config intact except for URL expansion, mirroring Agent.__init__.
        agent = Agent.from_dict(copy.deepcopy(self.agent_config))
        if not isinstance(agent, SimplifiedReActCodeAgent):
            raise TypeError(
                "counterfactual replay currently supports simplified_react_code_agent only; "
                f"got {type(agent).__name__}."
            )
        return agent

    def solve_tasks(
        self,
        task_ids: list[str],
        experiment_name: str,
        num_processes: int = 1,
        process_index: int = 0,
    ) -> None:
        self._experiment_name = experiment_name
        num_tasks = len(task_ids)
        num_processes = min(num_processes, num_tasks)
        task_ids = chunk_and_return(
            task_ids, num_chunks=num_processes, chunk_index=process_index, balanced=True
        )
        # Use a short-lived agent only for the standard AppWorld logger header/progress output.
        progress_agent = self._make_agent()
        progress_agent.logger.initialize(
            experiment_name=experiment_name,
            num_tasks=len(task_ids),
            num_processes=num_processes,
            process_index=process_index,
            extra_experiment_info={"Runner": "counterfactual_replay"},
        )
        appworld_config = copy.deepcopy(self._agent_appworld_config)
        with AppWorld.initializer(
            update_defaults=True, experiment_name=experiment_name, **appworld_config
        ):
            for task_id in task_ids:
                self.solve_task(task_id=task_id, progress_agent=progress_agent)
                self._set_finished(experiment_name, task_id)
                progress_agent.logger.complete_task()

    def solve_task(self, task_id: str, progress_agent: Agent | None = None) -> None:
        task_seed = self._stable_task_seed(task_id)
        task_random = random.Random(task_seed)
        root = self._run_trajectory(
            task_id=task_id,
            trajectory_id="root",
            case="root",
            attempt_index=0,
            parent_trajectory=None,
            modified_timestep=None,
            modification_kind=None,
            task_random=task_random,
        )
        trajectories = [root]
        root_success = bool(root.evaluation.get("success", False))
        final_success_id: str | None = root.trajectory_id if root_success else None
        final_failure_id: str | None = None if root_success else root.trajectory_id
        case = "B" if root_success else "A"

        if root_success:
            # Case B: perturb random timesteps until the replay fails. Save true->true attempts too.
            candidate_indices = list(range(len(root.steps)))
            for attempt_index in range(1, self.case_b_max_attempts + 1):
                if not candidate_indices:
                    break
                timestep = task_random.choice(candidate_indices)
                attempt = self._run_trajectory(
                    task_id=task_id,
                    trajectory_id=f"case_b_attempt_{attempt_index:03d}_t{timestep:03d}",
                    case="B",
                    attempt_index=attempt_index,
                    parent_trajectory=root,
                    modified_timestep=timestep,
                    modification_kind="plausible_wrong",
                    task_random=task_random,
                )
                trajectories.append(attempt)
                if not bool(attempt.evaluation.get("success", False)):
                    final_failure_id = attempt.trajectory_id
                    break
        else:
            # Case A: correct from the end backwards until the replay succeeds. Save false->false attempts too.
            attempt_index = 0
            max_attempts = min(self.max_replay_attempts, len(root.steps))
            for timestep in reversed(range(len(root.steps))):
                if attempt_index >= max_attempts:
                    break
                attempt_index += 1
                attempt = self._run_trajectory(
                    task_id=task_id,
                    trajectory_id=f"case_a_attempt_{attempt_index:03d}_t{timestep:03d}",
                    case="A",
                    attempt_index=attempt_index,
                    parent_trajectory=root,
                    modified_timestep=timestep,
                    modification_kind="corrective",
                    task_random=task_random,
                )
                trajectories.append(attempt)
                if bool(attempt.evaluation.get("success", False)):
                    final_success_id = attempt.trajectory_id
                    break

        for trajectory in trajectories:
            trajectory.is_final_pair_member = trajectory.trajectory_id in {
                final_success_id,
                final_failure_id,
            }
            self._save_trajectory(trajectory)

        manifest = {
            "task_id": task_id,
            "case": case,
            "root_success": root_success,
            "final_success_trajectory_id": final_success_id,
            "final_failure_trajectory_id": final_failure_id,
            "pair_complete": bool(final_success_id and final_failure_id),
            "trajectory_ids": [trajectory.trajectory_id for trajectory in trajectories],
            "num_trajectories": len(trajectories),
            "counterfactual_config": self.counterfactual_config,
        }
        self._write_task_counterfactual_json(task_id, "manifest.json", manifest)

    def _run_trajectory(
        self,
        task_id: str,
        trajectory_id: str,
        case: str,
        attempt_index: int,
        parent_trajectory: TrajectoryRecord | None,
        modified_timestep: int | None,
        modification_kind: str | None,
        task_random: random.Random,
    ) -> TrajectoryRecord:
        agent = self._make_agent()
        if self.random_seed is not None:
            set_random_seed(int(self.random_seed) + attempt_index)
        original_action: str | None = None
        modified_action: str | None = None
        modification_prompt: str | None = None
        steps: list[StepRecord] = []
        error: str | None = None

        with AppWorld(task_id=task_id) as world:
            agent.initialize(world)
            execution_outputs: Sequence[ExecutionIO] = []
            prefix_steps: list[StepRecord] = []
            if parent_trajectory is not None:
                if modified_timestep is None:
                    raise ValueError("modified_timestep is required for replay attempts.")
                prefix_steps = parent_trajectory.steps[:modified_timestep]
                for parent_step in prefix_steps:
                    # Re-execute the exact original prefix to reconstruct the state. This is intentionally
                    # not an LLM call, so the original prompt/history is preserved verbatim up to the branch.
                    output = world.batch_execute([parent_step.action])[0]
                    execution_outputs = [ExecutionIO(content=output)]
                agent.messages = copy.deepcopy(parent_trajectory.steps[modified_timestep].messages_before_action)
                steps.extend(copy.deepcopy(prefix_steps))
                original_action = parent_trajectory.steps[modified_timestep].action
                modification_prompt = self._modification_prompt(
                    kind=cast(str, modification_kind),
                    original_action=original_action,
                    task_random=task_random,
                )

            start_step = len(prefix_steps)
            for step_index in range(start_step, agent.max_steps):
                agent.step_number = step_index + 1
                try:
                    messages_before_action = copy.deepcopy(agent.messages)
                    prompt_for_this_action = (
                        modification_prompt if step_index == modified_timestep else None
                    )
                    action, assistant_message, usage, status = self._generate_action(
                        agent=agent, extra_action_instruction=prompt_for_this_action
                    )
                    if (
                        self.ensure_different_action
                        and step_index == modified_timestep
                        and original_action is not None
                        and action.strip() == original_action.strip()
                        and prompt_for_this_action is not None
                    ):
                        # The branch point must actually change the selected action. If the model
                        # repeats the original action, restore the exact pre-action history and retry
                        # the one modified-action prompt with a stricter local instruction.
                        agent.messages = copy.deepcopy(messages_before_action)
                        prompt_for_this_action = (
                            prompt_for_this_action
                            + "\n\nThe replacement must differ from the original action while remaining plausible."
                        )
                        modification_prompt = prompt_for_this_action
                        action, assistant_message, usage, status = self._generate_action(
                            agent=agent, extra_action_instruction=prompt_for_this_action
                        )
                    if status.failed:
                        error = status.message
                        break
                    if step_index == modified_timestep:
                        modified_action = action
                    execution_outputs_raw = world.batch_execute([action])
                    execution_outputs = [ExecutionIO(content=execution_outputs_raw[0])]
                    steps.append(
                        StepRecord(
                            step_index=step_index,
                            step_number=step_index + 1,
                            messages_before_action=messages_before_action,
                            assistant_message=assistant_message,
                            action=action,
                            environment_output=execution_outputs_raw[0],
                            usage=usage.dict(),
                            status={"failed": status.failed, "message": status.message},
                        )
                    )
                    agent.usage_tracker.add(task_id, usage)
                    self._append_environment_output(agent, execution_outputs[0].content)
                    if world.task_completed() or agent.usage_tracker.exceeded(task_id):
                        break
                except Exception as exception:  # Save failed/partial trajectories for debugging.
                    error = f"{type(exception).__name__}: {exception}"
                    break
            evaluation = self._evaluate(world)
            if error is not None:
                evaluation = {**evaluation, "runner_error": error}
            record = TrajectoryRecord(
                trajectory_id=trajectory_id,
                task_id=task_id,
                case=case,
                attempt_index=attempt_index,
                parent_trajectory_id=parent_trajectory.trajectory_id if parent_trajectory else None,
                source_trajectory_id=parent_trajectory.trajectory_id if parent_trajectory else None,
                modified_timestep=modified_timestep,
                original_action=original_action,
                modified_action=modified_action,
                modification_prompt=modification_prompt,
                action_changed=(
                    original_action is not None
                    and modified_action is not None
                    and original_action.strip() != modified_action.strip()
                ),
                evaluation=evaluation,
                steps=steps,
                messages=copy.deepcopy(agent.messages),
                environment_io=copy.deepcopy(world.environment_io),
                replay_objective_met=self._objective_met(case, evaluation),
                error=error,
            )
            if self.copy_attempt_dbs:
                self._copy_attempt_dbs(world, task_id, trajectory_id)
            return record

    def _generate_action(
        self, agent: SimplifiedReActCodeAgent, extra_action_instruction: str | None = None
    ) -> tuple[str, dict[str, Any], Usage, Status]:
        messages = copy.deepcopy(agent.trimmed_messages)
        if extra_action_instruction:
            messages.append({"role": "user", "content": extra_action_instruction})
        output = agent.language_model.generate(messages=messages, cache_control_at=-1)
        error_message = output.pop("error", None)
        if error_message:
            return "", {}, Usage(), Status(failed=True, message=error_message)
        usage = output.pop("standardized_usage")
        raw_message = copy.deepcopy(output)
        code, fixed_output_content = agent.extract_code_and_fix_content(output["content"] or "")
        output["content"] = fixed_output_content + "\n\n"
        # Important: do NOT append extra_action_instruction. Future steps see only the original
        # prompt/history plus the modified assistant action and its environment output.
        agent.messages.append(output)
        return code, raw_message, usage, Status(failed=False)

    def _append_environment_output(self, agent: SimplifiedReActCodeAgent, output: str) -> None:
        maybe_new_line = "\n" if not output.endswith("\n") else ""
        content = "Output:\n```\n" + output + maybe_new_line + "```\n\n"
        agent.messages.append({"role": "user", "content": content})

    def _modification_prompt(
        self, kind: str, original_action: str, task_random: random.Random
    ) -> str:
        nonce = task_random.randint(0, 10**9)
        if kind == "corrective":
            return (
                "For this next action only, replace the originally generated action below with a "
                "better action that is more likely to complete the task successfully. Keep the same "
                "overall task, prompt, style, and code-block output format. Do not mention this "
                "instruction. Return exactly one Python code block.\n\n"
                f"Original action to replace:\n```python\n{original_action.strip()}\n```\n"
                f"Correction nonce: {nonce}"
            )
        if kind == "plausible_wrong":
            return (
                "For this next action only, replace the originally generated action below with a "
                "plausible but subtly incorrect action that a model could naturally choose at this "
                "point. It should not look intentionally bad, malicious, or like sabotage; it should "
                "look like a reasonable but mistaken next step. Keep the same overall task, prompt, "
                "style, and code-block output format. Do not mention this instruction or that the "
                "action is wrong. Return exactly one Python code block.\n\n"
                f"Original action to replace:\n```python\n{original_action.strip()}\n```\n"
                f"Perturbation nonce: {nonce}"
            )
        raise ValueError(f"Unknown modification kind: {kind}")

    def _evaluate(self, world: AppWorld) -> dict[str, Any]:
        try:
            return cast(dict[str, Any], world.evaluate(suppress_errors=True).to_dict())
        except Exception as exception:
            return {
                "success": False,
                "evaluation_error": f"{type(exception).__name__}: {exception}",
            }

    def _objective_met(self, case: str, evaluation: dict[str, Any]) -> bool:
        success = bool(evaluation.get("success", False))
        if case == "A":
            return success
        if case == "B":
            return not success
        return False

    def _save_trajectory(self, trajectory: TrajectoryRecord) -> None:
        self._write_task_counterfactual_json(
            trajectory.task_id,
            os.path.join("trajectories", f"{trajectory.trajectory_id}.json"),
            trajectory.to_dict(),
        )

    def _write_task_counterfactual_json(
        self, task_id: str, relative_path: str, data: dict[str, Any]
    ) -> None:
        if not self._experiment_name:
            raise RuntimeError("experiment_name is not set")
        file_path = os.path.join(
            path_store.experiment_outputs,
            self._experiment_name,
            "tasks",
            task_id,
            "counterfactual",
            relative_path,
        )
        maybe_create_parent_directory(file_path)
        write_json(data, file_path, silent=True)

    def _copy_attempt_dbs(self, world: AppWorld, task_id: str, trajectory_id: str) -> None:
        if not self._experiment_name:
            raise RuntimeError("experiment_name is not set")
        source = world.output_db_home_path_on_disk
        if not os.path.isdir(source):
            return
        destination = os.path.join(
            path_store.experiment_outputs,
            self._experiment_name,
            "tasks",
            task_id,
            "counterfactual",
            "dbs",
            trajectory_id,
        )
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(source, destination)

    def _stable_task_seed(self, task_id: str) -> int:
        base = int(self.random_seed or 0)
        return base + sum((index + 1) * ord(char) for index, char in enumerate(task_id))

    def _set_finished(self, experiment_name: str, task_id: str) -> None:
        file_path = os.path.join(
            path_store.experiment_outputs,
            experiment_name,
            "tasks",
            task_id,
            "misc",
            "finished",
        )
        write_file("", file_path)


def run_experiment(
    experiment_name: str,
    runner_config: dict[str, Any],
    task_id: str | None = None,
    num_processes: int = 1,
    process_index: int = 0,
) -> None:
    agent_config = runner_config.pop("agent")
    dataset_name = runner_config.pop("dataset")
    counterfactual_config = runner_config.pop("counterfactual", {})
    if runner_config:
        raise Exception(f"Unexpected keys in the runner config: {runner_config}")
    if task_id:
        task_ids = [task_id]
    else:
        task_ids = load_task_ids(dataset_name)
    for task_id_ in task_ids:
        Task.load(task_id=task_id_)
    runner = CounterfactualReplayRunner(
        agent_config=agent_config,
        dataset_name=dataset_name,
        counterfactual_config=counterfactual_config,
    )
    runner.solve_tasks(
        task_ids=task_ids,
        experiment_name=experiment_name,
        num_processes=num_processes,
        process_index=process_index,
    )
