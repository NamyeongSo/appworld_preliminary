from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Protocol

from appworld_agents.code.context_engineering.action import (
    ActionSignature,
    extract_action_signatures,
)
from appworld_agents.code.context_engineering.config import SmoothingConfig
from appworld_agents.code.context_engineering.delta import DeltaCfRecord


@dataclass(frozen=True)
class ActionSampleRecord:
    """One raw policy sample from a fixed counterfactual state."""

    sample_id: str
    task_id: str
    timestep: int | None
    policy_name: str
    sample_index: int
    random_seed: int | None
    state_snapshot: str
    action: str | None
    matched_success_action: bool
    action_signature: ActionSignature | None = None
    action_parse_error: str | None = None
    error: str | None = None

    @classmethod
    def from_action(
        cls,
        *,
        sample_id: str,
        task_id: str,
        timestep: int | None,
        policy_name: str,
        sample_index: int,
        random_seed: int | None,
        state_snapshot: str,
        action: str | None,
        success_action_signature: ActionSignature | None,
        action_match_level: str,
        error: str | None = None,
    ) -> ActionSampleRecord:
        action_parse_result = extract_action_signatures(action or "")
        action_signature = action_parse_result.primary_signature
        matched_success_action = False
        if success_action_signature is not None and action_signature is not None:
            matched_success_action = action_signature.match_key(
                action_match_level
            ) == success_action_signature.match_key(action_match_level)
        return cls(
            sample_id=sample_id,
            task_id=task_id,
            timestep=timestep,
            policy_name=policy_name,
            sample_index=sample_index,
            random_seed=random_seed,
            state_snapshot=state_snapshot,
            action=action,
            matched_success_action=matched_success_action,
            action_signature=action_signature,
            action_parse_error=action_parse_result.parsing_error,
            error=error,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "task_id": self.task_id,
            "timestep": self.timestep,
            "policy_name": self.policy_name,
            "sample_index": self.sample_index,
            "random_seed": self.random_seed,
            "state_snapshot": self.state_snapshot,
            "action": self.action,
            "matched_success_action": self.matched_success_action,
            "action_signature": (
                self.action_signature.to_dict() if self.action_signature is not None else None
            ),
            "action_parse_error": self.action_parse_error,
            "error": self.error,
        }


@dataclass(frozen=True)
class RatioEstimateRecord:
    """Smoothed estimate of pi_theta/pi_old for one counterfactual delta."""

    task_id: str
    timestep: int | None
    success_trajectory_id: str | None
    failure_trajectory_id: str | None
    success_action: str | None
    failure_action: str | None
    state_snapshot: str | None
    pi_old: float | None
    pi_theta: float | None
    ratio: float | None
    count_old: int
    count_theta: int
    n_old: int
    n_theta: int
    smoothing_k: float
    smoothing_alpha: float
    action_match_level: str
    success_action_signature: dict[str, Any] | None
    raw_samples_path: str | None = None
    estimation_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "timestep": self.timestep,
            "success_trajectory_id": self.success_trajectory_id,
            "failure_trajectory_id": self.failure_trajectory_id,
            "success_action": self.success_action,
            "failure_action": self.failure_action,
            "state_snapshot": self.state_snapshot,
            "pi_old": self.pi_old,
            "pi_theta": self.pi_theta,
            "ratio": self.ratio,
            "count_old": self.count_old,
            "count_theta": self.count_theta,
            "N_old": self.n_old,
            "N_theta": self.n_theta,
            "smoothing": {
                "K": self.smoothing_k,
                "alpha": self.smoothing_alpha,
            },
            "action_match_level": self.action_match_level,
            "success_action_signature": self.success_action_signature,
            "raw_samples_path": self.raw_samples_path,
            "estimation_error": self.estimation_error,
        }


class RatioActionSampler(Protocol):
    def sample_action(
        self,
        *,
        task_id: str,
        timestep: int | None,
        state_snapshot: str,
        messages_before_action: list[dict[str, Any]],
        policy_name: str,
        sample_index: int,
        random_seed: int | None,
        success_action_signature: ActionSignature | None,
        action_match_level: str,
    ) -> ActionSampleRecord: ...


class AppWorldRatioActionSampler:
    """Samples one ReAct code action from stored messages without executing AppWorld."""

    def __init__(self, agent_config: dict[str, Any]) -> None:
        self.agent_config = copy.deepcopy(agent_config)

    def sample_action(
        self,
        *,
        task_id: str,
        timestep: int | None,
        state_snapshot: str,
        messages_before_action: list[dict[str, Any]],
        policy_name: str,
        sample_index: int,
        random_seed: int | None,
        success_action_signature: ActionSignature | None,
        action_match_level: str,
    ) -> ActionSampleRecord:
        sample_id = build_ratio_sample_id(
            task_id=task_id,
            timestep=timestep,
            policy_name=policy_name,
            sample_index=sample_index,
        )
        try:
            if random_seed is not None:
                from appworld.common.random import set_random_seed

                set_random_seed(random_seed)
            agent = self._make_agent(random_seed=random_seed)
            agent.messages = copy.deepcopy(messages_before_action)
            output = agent.language_model.generate(
                messages=copy.deepcopy(messages_before_action), cache_control_at=-1
            )
            error_message = output.pop("error", None)
            if error_message:
                return ActionSampleRecord.from_action(
                    sample_id=sample_id,
                    task_id=task_id,
                    timestep=timestep,
                    policy_name=policy_name,
                    sample_index=sample_index,
                    random_seed=random_seed,
                    state_snapshot=state_snapshot,
                    action=None,
                    success_action_signature=success_action_signature,
                    action_match_level=action_match_level,
                    error=str(error_message),
                )
            action, _ = agent.extract_code_and_fix_content(output.get("content") or "")
            return ActionSampleRecord.from_action(
                sample_id=sample_id,
                task_id=task_id,
                timestep=timestep,
                policy_name=policy_name,
                sample_index=sample_index,
                random_seed=random_seed,
                state_snapshot=state_snapshot,
                action=action,
                success_action_signature=success_action_signature,
                action_match_level=action_match_level,
            )
        except Exception as exception:
            return ActionSampleRecord.from_action(
                sample_id=sample_id,
                task_id=task_id,
                timestep=timestep,
                policy_name=policy_name,
                sample_index=sample_index,
                random_seed=random_seed,
                state_snapshot=state_snapshot,
                action=None,
                success_action_signature=success_action_signature,
                action_match_level=action_match_level,
                error=f"{type(exception).__name__}: {exception}",
            )

    def _make_agent(self, random_seed: int | None = None) -> Any:
        from appworld_agents.code.simplified.agent import Agent
        from appworld_agents.code.simplified.react_code_agent import (
            SimplifiedReActCodeAgent,
        )

        agent = Agent.from_dict(self._agent_config_for_seed(random_seed))
        if not isinstance(agent, SimplifiedReActCodeAgent):
            raise TypeError(
                "Context Engineering Phase 5 currently supports simplified_react_code_agent "
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


@dataclass(frozen=True)
class RatioEstimationInputs:
    record: RatioEstimateRecord
    samples: list[ActionSampleRecord]


def estimate_ratio_from_samples(
    *,
    delta_record: DeltaCfRecord,
    old_samples: list[ActionSampleRecord],
    theta_samples: list[ActionSampleRecord],
    smoothing: SmoothingConfig,
    raw_samples_path: str | None = None,
) -> RatioEstimateRecord:
    success_action_signature = _success_action_signature(delta_record)
    if delta_record.extraction_error is not None:
        return _error_record(
            delta_record=delta_record,
            smoothing=smoothing,
            old_samples=old_samples,
            theta_samples=theta_samples,
            raw_samples_path=raw_samples_path,
            estimation_error=f"Cannot estimate ratio for delta extraction error: {delta_record.extraction_error}",
            success_action_signature=success_action_signature,
        )
    if delta_record.state_snapshot is None:
        return _error_record(
            delta_record=delta_record,
            smoothing=smoothing,
            old_samples=old_samples,
            theta_samples=theta_samples,
            raw_samples_path=raw_samples_path,
            estimation_error="Cannot estimate ratio without a state_snapshot.",
            success_action_signature=success_action_signature,
        )
    if success_action_signature is None:
        return _error_record(
            delta_record=delta_record,
            smoothing=smoothing,
            old_samples=old_samples,
            theta_samples=theta_samples,
            raw_samples_path=raw_samples_path,
            estimation_error="Cannot estimate ratio because success_action has no parseable signature.",
            success_action_signature=None,
        )

    count_old = _match_count(old_samples)
    count_theta = _match_count(theta_samples)
    n_old = len(old_samples)
    n_theta = len(theta_samples)
    pi_old = _smoothed_probability(count=count_old, sample_size=n_old, smoothing=smoothing)
    pi_theta = _smoothed_probability(count=count_theta, sample_size=n_theta, smoothing=smoothing)
    ratio = pi_theta / pi_old
    return RatioEstimateRecord(
        task_id=delta_record.task_id,
        timestep=delta_record.timestep,
        success_trajectory_id=delta_record.success_trajectory_id,
        failure_trajectory_id=delta_record.failure_trajectory_id,
        success_action=delta_record.success_action,
        failure_action=delta_record.failure_action,
        state_snapshot=delta_record.state_snapshot,
        pi_old=pi_old,
        pi_theta=pi_theta,
        ratio=ratio,
        count_old=count_old,
        count_theta=count_theta,
        n_old=n_old,
        n_theta=n_theta,
        smoothing_k=smoothing.k,
        smoothing_alpha=smoothing.alpha,
        action_match_level=smoothing.action_match_level,
        success_action_signature=success_action_signature.to_dict(),
        raw_samples_path=raw_samples_path,
    )


def build_ratio_sample_id(
    *, task_id: str, timestep: int | None, policy_name: str, sample_index: int
) -> str:
    safe_task_id = "".join(
        character if character.isalnum() or character in "_.-" else "_" for character in task_id
    )
    timestep_component = "none" if timestep is None else f"{timestep:03d}"
    return f"{safe_task_id}__t{timestep_component}__{policy_name}_{sample_index:03d}"


def messages_from_state_snapshot(state_snapshot: str) -> list[dict[str, Any]]:
    payload = json.loads(state_snapshot)
    if not isinstance(payload, list):
        raise ValueError("state_snapshot must encode a list of messages")
    return [dict(item) for item in payload]


def _success_action_signature(delta_record: DeltaCfRecord) -> ActionSignature | None:
    payload = delta_record.delta_cf.get("success_action_signature")
    if isinstance(payload, dict):
        argument_keys = payload.get("argument_keys", [])
        if isinstance(argument_keys, list):
            return ActionSignature(
                app_name=str(payload["app_name"]),
                api_name=str(payload["api_name"]),
                argument_keys=tuple(str(key) for key in argument_keys),
            )
    if delta_record.success_action is None:
        return None
    return extract_action_signatures(delta_record.success_action).primary_signature


def _smoothed_probability(*, count: int, sample_size: int, smoothing: SmoothingConfig) -> float:
    return (count + smoothing.alpha) / (sample_size + smoothing.k * smoothing.alpha)


def _match_count(samples: list[ActionSampleRecord]) -> int:
    return sum(1 for sample in samples if sample.matched_success_action)


def _error_record(
    *,
    delta_record: DeltaCfRecord,
    smoothing: SmoothingConfig,
    old_samples: list[ActionSampleRecord],
    theta_samples: list[ActionSampleRecord],
    raw_samples_path: str | None,
    estimation_error: str,
    success_action_signature: ActionSignature | None,
) -> RatioEstimateRecord:
    return RatioEstimateRecord(
        task_id=delta_record.task_id,
        timestep=delta_record.timestep,
        success_trajectory_id=delta_record.success_trajectory_id,
        failure_trajectory_id=delta_record.failure_trajectory_id,
        success_action=delta_record.success_action,
        failure_action=delta_record.failure_action,
        state_snapshot=delta_record.state_snapshot,
        pi_old=None,
        pi_theta=None,
        ratio=None,
        count_old=_match_count(old_samples),
        count_theta=_match_count(theta_samples),
        n_old=len(old_samples),
        n_theta=len(theta_samples),
        smoothing_k=smoothing.k,
        smoothing_alpha=smoothing.alpha,
        action_match_level=smoothing.action_match_level,
        success_action_signature=(
            success_action_signature.to_dict() if success_action_signature is not None else None
        ),
        raw_samples_path=raw_samples_path,
        estimation_error=estimation_error,
    )
