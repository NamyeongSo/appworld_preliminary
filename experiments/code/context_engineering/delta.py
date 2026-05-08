from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from appworld_agents.code.context_engineering.action import (
    ActionSignature,
    extract_action_signatures,
    render_state_snapshot,
)
from appworld_agents.code.context_engineering.counterfactual import (
    CounterfactualPairRecord,
    CounterfactualTrajectoryRecord,
)


@dataclass(frozen=True)
class DeltaCfRecord:
    task_id: str
    timestep: int | None
    success_trajectory_id: str | None
    failure_trajectory_id: str | None
    success_action: str | None
    failure_action: str | None
    delta_cf: dict[str, Any]
    state_snapshot: str | None = None
    messages_before_action: list[dict[str, Any]] | None = None
    state_source_trajectory_id: str | None = None
    extraction_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "timestep": self.timestep,
            "success_trajectory_id": self.success_trajectory_id,
            "failure_trajectory_id": self.failure_trajectory_id,
            "success_action": self.success_action,
            "failure_action": self.failure_action,
            "state_snapshot": self.state_snapshot,
            "messages_before_action": self.messages_before_action,
            "state_source_trajectory_id": self.state_source_trajectory_id,
            "delta_cf": self.delta_cf,
            "extraction_error": self.extraction_error,
        }


def extract_delta_cf_records(
    pair_record: CounterfactualPairRecord,
) -> list[DeltaCfRecord]:
    """Extract success-vs-failure action deltas from one counterfactual pair."""

    if not pair_record.pair_complete:
        return [
            _error_record(
                pair_record=pair_record,
                extraction_error="Counterfactual pair is incomplete.",
            )
        ]

    success_trajectory = _trajectory_by_id(pair_record, pair_record.final_success_trajectory_id)
    failure_trajectory = _trajectory_by_id(pair_record, pair_record.final_failure_trajectory_id)
    if success_trajectory is None or failure_trajectory is None:
        return [
            _error_record(
                pair_record=pair_record,
                extraction_error=(
                    "Final success/failure trajectory ids are not present in the trajectory set."
                ),
            )
        ]

    branch_trajectory = _branch_trajectory(success_trajectory, failure_trajectory)
    if branch_trajectory is None:
        return [
            _error_record(
                pair_record=pair_record,
                extraction_error="No final pair trajectory records a modified timestep.",
            )
        ]

    if branch_trajectory.modified_timestep is None:
        return [
            _error_record(
                pair_record=pair_record,
                extraction_error="Branch trajectory has no modified timestep.",
            )
        ]

    if branch_trajectory.original_action is None or branch_trajectory.modified_action is None:
        return [
            _error_record(
                pair_record=pair_record,
                extraction_error=(
                    "Branch trajectory is missing original_action or modified_action."
                ),
            )
        ]

    if branch_trajectory.trajectory_id == success_trajectory.trajectory_id:
        success_action = branch_trajectory.modified_action
        failure_action = branch_trajectory.original_action
    else:
        success_action = branch_trajectory.original_action
        failure_action = branch_trajectory.modified_action
    state_source_trajectory = (
        _trajectory_by_id(pair_record, branch_trajectory.source_trajectory_id)
        or _trajectory_by_id(pair_record, branch_trajectory.parent_trajectory_id)
        or branch_trajectory
    )
    state_step = _step_at_timestep(
        trajectory=state_source_trajectory,
        timestep=branch_trajectory.modified_timestep,
    )
    state_snapshot, messages_before_action = _state_from_step(state_step)

    return [
        DeltaCfRecord(
            task_id=pair_record.task_id,
            timestep=branch_trajectory.modified_timestep,
            success_trajectory_id=success_trajectory.trajectory_id,
            failure_trajectory_id=failure_trajectory.trajectory_id,
            success_action=success_action,
            failure_action=failure_action,
            state_snapshot=state_snapshot,
            messages_before_action=messages_before_action,
            state_source_trajectory_id=(
                state_source_trajectory.trajectory_id
                if state_source_trajectory is not None
                else None
            ),
            delta_cf=_build_delta_cf(success_action, failure_action),
        )
    ]


def _build_delta_cf(success_action: str, failure_action: str) -> dict[str, Any]:
    success_parse_result = extract_action_signatures(success_action)
    failure_parse_result = extract_action_signatures(failure_action)
    success_signature = success_parse_result.primary_signature
    failure_signature = failure_parse_result.primary_signature
    api_difference = _api_difference(success_signature, failure_signature)
    argument_difference = _argument_difference(success_signature, failure_signature)
    call_sequence_difference = _call_sequence_difference(
        success_parse_result.signatures, failure_parse_result.signatures
    )
    return {
        "api_difference": api_difference,
        "argument_difference": argument_difference,
        "call_sequence_difference": call_sequence_difference,
        "success_action_signature": (
            success_signature.to_dict() if success_signature is not None else None
        ),
        "failure_action_signature": (
            failure_signature.to_dict() if failure_signature is not None else None
        ),
        "success_action_parse_error": success_parse_result.parsing_error,
        "failure_action_parse_error": failure_parse_result.parsing_error,
        "summary": _diff_summary(
            api_difference=api_difference,
            argument_difference=argument_difference,
            call_sequence_difference=call_sequence_difference,
        ),
    }


def _api_difference(
    success_signature: ActionSignature | None,
    failure_signature: ActionSignature | None,
) -> dict[str, Any]:
    success_api = success_signature.full_api_name if success_signature is not None else None
    failure_api = failure_signature.full_api_name if failure_signature is not None else None
    return {
        "success_api": success_api,
        "failure_api": failure_api,
        "changed": success_api != failure_api,
    }


def _argument_difference(
    success_signature: ActionSignature | None,
    failure_signature: ActionSignature | None,
) -> dict[str, Any]:
    success_keys = set(success_signature.argument_keys if success_signature else ())
    failure_keys = set(failure_signature.argument_keys if failure_signature else ())
    return {
        "success_argument_keys": sorted(success_keys),
        "failure_argument_keys": sorted(failure_keys),
        "added_in_success": sorted(success_keys - failure_keys),
        "missing_in_success": sorted(failure_keys - success_keys),
        "changed": success_keys != failure_keys,
    }


def _call_sequence_difference(
    success_signatures: tuple[ActionSignature, ...],
    failure_signatures: tuple[ActionSignature, ...],
) -> dict[str, Any]:
    success_sequence = [signature.full_api_name for signature in success_signatures]
    failure_sequence = [signature.full_api_name for signature in failure_signatures]
    return {
        "success_api_sequence": success_sequence,
        "failure_api_sequence": failure_sequence,
        "changed": success_sequence != failure_sequence,
    }


def _diff_summary(
    *,
    api_difference: dict[str, Any],
    argument_difference: dict[str, Any],
    call_sequence_difference: dict[str, Any],
) -> str:
    parts: list[str] = []
    if api_difference["changed"]:
        parts.append(
            f"API changed from {api_difference['failure_api']} to {api_difference['success_api']}."
        )
    else:
        parts.append(f"API stayed at {api_difference['success_api']}.")
    if argument_difference["changed"]:
        parts.append(
            "Argument keys changed; added in success="
            f"{argument_difference['added_in_success']}, missing in success="
            f"{argument_difference['missing_in_success']}."
        )
    else:
        parts.append("Argument key set stayed the same.")
    if call_sequence_difference["changed"]:
        parts.append("API call order/sequence changed.")
    else:
        parts.append("API call order/sequence stayed the same.")
    return " ".join(parts)


def _branch_trajectory(
    success_trajectory: CounterfactualTrajectoryRecord,
    failure_trajectory: CounterfactualTrajectoryRecord,
) -> CounterfactualTrajectoryRecord | None:
    if success_trajectory.modified_timestep is not None:
        return success_trajectory
    if failure_trajectory.modified_timestep is not None:
        return failure_trajectory
    return None


def _trajectory_by_id(
    pair_record: CounterfactualPairRecord, trajectory_id: str | None
) -> CounterfactualTrajectoryRecord | None:
    if trajectory_id is None:
        return None
    for trajectory in pair_record.trajectories:
        if trajectory.trajectory_id == trajectory_id:
            return trajectory
    return None


def _step_at_timestep(
    *,
    trajectory: CounterfactualTrajectoryRecord | None,
    timestep: int,
) -> dict[str, Any] | None:
    if trajectory is None:
        return None
    for step in trajectory.steps:
        if step.get("step_index") == timestep or step.get("timestep") == timestep:
            return step
    if 0 <= timestep < len(trajectory.steps):
        return trajectory.steps[timestep]
    return None


def _state_from_step(
    step: dict[str, Any] | None,
) -> tuple[str | None, list[dict[str, Any]] | None]:
    if step is None:
        return None, None
    messages_before_action = step.get("messages_before_action")
    if isinstance(messages_before_action, list):
        messages = [
            dict(message) for message in messages_before_action if isinstance(message, dict)
        ]
    else:
        messages = None
    state_snapshot = step.get("state_snapshot")
    if isinstance(state_snapshot, str):
        return state_snapshot, messages
    if messages is None:
        return None, None
    return render_state_snapshot(messages), messages


def _error_record(
    *,
    pair_record: CounterfactualPairRecord,
    extraction_error: str,
) -> DeltaCfRecord:
    return DeltaCfRecord(
        task_id=pair_record.task_id,
        timestep=None,
        success_trajectory_id=pair_record.final_success_trajectory_id,
        failure_trajectory_id=pair_record.final_failure_trajectory_id,
        success_action=None,
        failure_action=None,
        delta_cf={},
        extraction_error=extraction_error,
    )
