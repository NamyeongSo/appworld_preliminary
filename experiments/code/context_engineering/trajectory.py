from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from appworld_agents.code.context_engineering.action import build_step_representation


@dataclass(frozen=True)
class BaselineStepRecord:
    step_index: int
    step_number: int
    messages_before_action: list[dict[str, Any]]
    assistant_message: dict[str, Any]
    action: str
    environment_output: str
    usage: dict[str, Any]
    status: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        step_representation = build_step_representation(
            messages_before_action=self.messages_before_action,
            action=self.action,
        )
        return {
            "step_index": self.step_index,
            "step_number": self.step_number,
            "messages_before_action": self.messages_before_action,
            "assistant_message": self.assistant_message,
            "action": self.action,
            "environment_output": self.environment_output,
            "usage": self.usage,
            "status": self.status,
            **step_representation.to_dict(),
        }


@dataclass(frozen=True)
class BaselineTrajectoryRecord:
    trajectory_id: str
    task_id: str
    loop_index: int
    sample_index: int
    random_seed: int | None
    evaluation: dict[str, Any]
    steps: list[BaselineStepRecord]
    messages: list[dict[str, Any]]
    environment_io: list[dict[str, Any]]
    error: str | None = None

    @property
    def success(self) -> bool:
        return bool(self.evaluation.get("success", False))

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "task_id": self.task_id,
            "loop_index": self.loop_index,
            "sample_index": self.sample_index,
            "random_seed": self.random_seed,
            "evaluation": self.evaluation,
            "success": self.success,
            "error": self.error,
            "num_steps": len(self.steps),
            "steps": [step.to_dict() for step in self.steps],
            "messages": self.messages,
            "environment_io": self.environment_io,
        }


class BaselineTrajectoryExecutor(Protocol):
    def run_trajectory(
        self,
        *,
        task_id: str,
        trajectory_id: str,
        loop_index: int,
        sample_index: int,
        random_seed: int | None,
    ) -> BaselineTrajectoryRecord: ...


class AppWorldBaselineTrajectoryExecutor:
    def __init__(self, agent_config: dict[str, Any]) -> None:
        self.agent_config = copy.deepcopy(agent_config)

    def run_trajectory(
        self,
        *,
        task_id: str,
        trajectory_id: str,
        loop_index: int,
        sample_index: int,
        random_seed: int | None,
    ) -> BaselineTrajectoryRecord:
        # AppWorld is optional until users run `appworld install --repo`, so import it only on
        # the real execution path. Unit tests inject a fake executor and do not need this import.
        from appworld import AppWorld
        from appworld_agents.code.simplified.agent import ExecutionIO

        agent = self._make_agent(random_seed=random_seed)
        if random_seed is not None:
            from appworld.common.random import set_random_seed

            set_random_seed(random_seed)

        steps: list[BaselineStepRecord] = []
        error: str | None = None
        with AppWorld(task_id=task_id) as world:
            agent.initialize(world)
            execution_outputs: Sequence[ExecutionIO] = []
            for step_index in range(agent.max_steps):
                agent.step_number = step_index + 1
                try:
                    messages_before_action = copy.deepcopy(agent.messages)
                    action, assistant_message, usage, status = self._generate_action(agent=agent)
                    if status.failed:
                        error = status.message
                        break
                    execution_outputs_raw = world.batch_execute([action])
                    execution_outputs = [ExecutionIO(content=execution_outputs_raw[0])]
                    steps.append(
                        BaselineStepRecord(
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
                    self._append_environment_output(
                        agent=agent, output=execution_outputs[0].content
                    )
                    if world.task_completed() or agent.usage_tracker.exceeded(task_id):
                        break
                except Exception as exception:
                    error = f"{type(exception).__name__}: {exception}"
                    break
            evaluation = self._evaluate(world)
            if error is not None:
                evaluation = {**evaluation, "runner_error": error}
            return BaselineTrajectoryRecord(
                trajectory_id=trajectory_id,
                task_id=task_id,
                loop_index=loop_index,
                sample_index=sample_index,
                random_seed=random_seed,
                evaluation=evaluation,
                steps=steps,
                messages=copy.deepcopy(agent.messages),
                environment_io=copy.deepcopy(world.environment_io),
                error=error,
            )

    def _make_agent(self, random_seed: int | None = None) -> Any:
        from appworld_agents.code.simplified.agent import Agent
        from appworld_agents.code.simplified.react_code_agent import (
            SimplifiedReActCodeAgent,
        )

        agent = Agent.from_dict(self._agent_config_for_seed(random_seed))
        if not isinstance(agent, SimplifiedReActCodeAgent):
            raise TypeError(
                "Context Engineering Phase 1 currently supports simplified_react_code_agent "
                f"only; got {type(agent).__name__}."
            )
        return agent

    def _agent_config_for_seed(self, random_seed: int | None) -> dict[str, Any]:
        agent_config = copy.deepcopy(self.agent_config)
        if random_seed is None:
            return agent_config
        appworld_config = dict(agent_config.get("appworld_config", {}) or {})
        appworld_config["random_seed"] = random_seed
        agent_config["appworld_config"] = appworld_config
        model_config = dict(agent_config.get("model_config", {}) or {})
        if "seed" in model_config:
            model_config["seed"] = random_seed
            agent_config["model_config"] = model_config
        return agent_config

    def _generate_action(self, agent: Any) -> tuple[str, dict[str, Any], Any, Any]:
        from appworld_agents.code.common.usage_tracker import Usage
        from appworld_agents.code.simplified.agent import Status

        messages = copy.deepcopy(agent.trimmed_messages)
        output = agent.language_model.generate(messages=messages, cache_control_at=-1)
        error_message = output.pop("error", None)
        if error_message:
            return "", {}, Usage(), Status(failed=True, message=error_message)
        usage = output.pop("standardized_usage")
        raw_message = copy.deepcopy(output)
        code, fixed_output_content = agent.extract_code_and_fix_content(output["content"] or "")
        output["content"] = fixed_output_content + "\n\n"
        agent.messages.append(output)
        return code, raw_message, usage, Status(failed=False)

    def _append_environment_output(self, agent: Any, output: str) -> None:
        maybe_new_line = "\n" if not output.endswith("\n") else ""
        content = "Output:\n```\n" + output + maybe_new_line + "```\n\n"
        agent.messages.append({"role": "user", "content": content})

    def _evaluate(self, world: Any) -> dict[str, Any]:
        try:
            return cast(dict[str, Any], world.evaluate(suppress_errors=True).to_dict())
        except Exception as exception:
            return {
                "success": False,
                "evaluation_error": f"{type(exception).__name__}: {exception}",
            }
