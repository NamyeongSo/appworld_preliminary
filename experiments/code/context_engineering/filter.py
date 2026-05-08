from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from appworld_agents.code.context_engineering.config import RatioFilterConfig


@dataclass(frozen=True)
class RatioFilterDecisionRecord:
    """Phase 6 bucket assignment for one ratio estimate."""

    sample_id: str
    task_id: str
    timestep: int | None
    bucket: str
    ratio: float | None
    lower_bound: float
    upper_bound: float
    defer_round: int
    next_loop_index: int | None
    reason: str
    ratio_record: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "task_id": self.task_id,
            "timestep": self.timestep,
            "bucket": self.bucket,
            "ratio": self.ratio,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "defer_round": self.defer_round,
            "next_loop_index": self.next_loop_index,
            "reason": self.reason,
            "ratio_record": self.ratio_record,
        }


def filter_ratio_records(
    *,
    ratio_records: list[dict[str, Any]],
    ratio_filter_config: RatioFilterConfig,
    loop_index: int,
) -> dict[str, list[RatioFilterDecisionRecord]]:
    """Classify each ratio record into selected, deferred, or discarded exactly once."""

    selected_records: list[RatioFilterDecisionRecord] = []
    deferred_records: list[RatioFilterDecisionRecord] = []
    discarded_records: list[RatioFilterDecisionRecord] = []
    for ratio_index, ratio_record in enumerate(ratio_records):
        decision = classify_ratio_record(
            ratio_record=ratio_record,
            ratio_filter_config=ratio_filter_config,
            loop_index=loop_index,
            ratio_index=ratio_index,
        )
        if decision.bucket == "selected":
            selected_records.append(decision)
        elif decision.bucket == "deferred":
            deferred_records.append(decision)
        elif decision.bucket == "discarded":
            discarded_records.append(decision)
        else:  # defensive guard for future bucket names.
            raise ValueError(f"Unsupported ratio bucket: {decision.bucket!r}")
    return {
        "selected": selected_records,
        "deferred": deferred_records,
        "discarded": discarded_records,
    }


def classify_ratio_record(
    *,
    ratio_record: dict[str, Any],
    ratio_filter_config: RatioFilterConfig,
    loop_index: int,
    ratio_index: int = 0,
) -> RatioFilterDecisionRecord:
    lower_bound = 1.0 - ratio_filter_config.epsilon
    upper_bound = 1.0 + ratio_filter_config.epsilon
    ratio = _optional_float(ratio_record.get("ratio"))
    prior_defer_round = _prior_defer_round(ratio_record)
    sample_id = _sample_id(ratio_record=ratio_record, ratio_index=ratio_index)

    if ratio is None:
        return RatioFilterDecisionRecord(
            sample_id=sample_id,
            task_id=str(ratio_record.get("task_id", "")),
            timestep=_optional_int(ratio_record.get("timestep")),
            bucket="discarded",
            ratio=None,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            defer_round=prior_defer_round,
            next_loop_index=None,
            reason=_ratio_error_reason(ratio_record),
            ratio_record=ratio_record,
        )

    if lower_bound <= ratio <= upper_bound:
        return RatioFilterDecisionRecord(
            sample_id=sample_id,
            task_id=str(ratio_record.get("task_id", "")),
            timestep=_optional_int(ratio_record.get("timestep")),
            bucket="selected",
            ratio=ratio,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            defer_round=prior_defer_round,
            next_loop_index=None,
            reason="ratio_within_acceptance_band",
            ratio_record=ratio_record,
        )

    if ratio > upper_bound:
        next_defer_round = prior_defer_round + 1
        if next_defer_round > ratio_filter_config.max_defer_rounds:
            return RatioFilterDecisionRecord(
                sample_id=sample_id,
                task_id=str(ratio_record.get("task_id", "")),
                timestep=_optional_int(ratio_record.get("timestep")),
                bucket="discarded",
                ratio=ratio,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                defer_round=next_defer_round,
                next_loop_index=None,
                reason="max_defer_rounds_exceeded",
                ratio_record=ratio_record,
            )
        return RatioFilterDecisionRecord(
            sample_id=sample_id,
            task_id=str(ratio_record.get("task_id", "")),
            timestep=_optional_int(ratio_record.get("timestep")),
            bucket="deferred",
            ratio=ratio,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            defer_round=next_defer_round,
            next_loop_index=loop_index + 1,
            reason="ratio_above_acceptance_band",
            ratio_record=ratio_record,
        )

    return RatioFilterDecisionRecord(
        sample_id=sample_id,
        task_id=str(ratio_record.get("task_id", "")),
        timestep=_optional_int(ratio_record.get("timestep")),
        bucket="discarded",
        ratio=ratio,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        defer_round=prior_defer_round,
        next_loop_index=None,
        reason="ratio_below_acceptance_band",
        ratio_record=ratio_record,
    )


def _sample_id(*, ratio_record: dict[str, Any], ratio_index: int) -> str:
    task_id = str(ratio_record.get("task_id", "unknown"))
    timestep = ratio_record.get("timestep")
    success_trajectory_id = ratio_record.get("success_trajectory_id") or "none"
    failure_trajectory_id = ratio_record.get("failure_trajectory_id") or "none"
    return "::".join(
        [
            task_id,
            f"t={timestep}",
            str(success_trajectory_id),
            str(failure_trajectory_id),
            str(ratio_index),
        ]
    )


def _prior_defer_round(ratio_record: dict[str, Any]) -> int:
    for key in ("defer_round", "prior_defer_round"):
        value = ratio_record.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return 0


def _ratio_error_reason(ratio_record: dict[str, Any]) -> str:
    estimation_error = ratio_record.get("estimation_error")
    if estimation_error:
        return f"ratio_estimation_error: {estimation_error}"
    return "ratio_missing"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
