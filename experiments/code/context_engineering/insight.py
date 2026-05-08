from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class InsightRecord:
    """Evidence-backed policy insight extracted from one selected CE sample."""

    task_id: str
    timestep: int | None
    state_summary: str
    success_action: str | None
    failure_action: str | None
    delta_cf: dict[str, Any]
    insight: str
    ratio: float | None
    source: str
    selected_sample_id: str
    evidence: dict[str, Any]
    validation_flags: list[str]
    rejected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "timestep": self.timestep,
            "state_summary": self.state_summary,
            "success_action": self.success_action,
            "failure_action": self.failure_action,
            "delta_cf": self.delta_cf,
            "insight": self.insight,
            "ratio": self.ratio,
            "source": self.source,
            "selected_sample_id": self.selected_sample_id,
            "evidence": self.evidence,
            "validation_flags": self.validation_flags,
            "rejected": self.rejected,
        }


class InsightExtractor(Protocol):
    def extract_insight(self, *, selected_sample: dict[str, Any]) -> InsightRecord: ...


class HeuristicInsightExtractor:
    """Deterministic Phase 7 extractor until an LLM-backed extractor is added."""

    def extract_insight(self, *, selected_sample: dict[str, Any]) -> InsightRecord:
        ratio_record = _ratio_record(selected_sample)
        delta_cf = _dict_value(ratio_record.get("delta_cf"))
        success_action = _optional_str(ratio_record.get("success_action"))
        failure_action = _optional_str(ratio_record.get("failure_action"))
        success_signature = _dict_value(ratio_record.get("success_action_signature"))
        if not success_signature:
            success_signature = _dict_value(delta_cf.get("success_action_signature"))
        failure_signature = _dict_value(delta_cf.get("failure_action_signature"))
        success_api = _api_name(success_signature, success_action)
        failure_api = _api_name(failure_signature, failure_action)
        argument_keys = _argument_keys(success_signature)
        delta_summary = str(delta_cf.get("summary") or "No delta summary recorded.")
        state_summary = summarize_state(ratio_record.get("state_snapshot"))
        insight = _build_insight_text(
            state_summary=state_summary,
            success_api=success_api,
            argument_keys=argument_keys,
            failure_api=failure_api,
            delta_summary=delta_summary,
        )
        validation_flags = _validation_flags(
            insight=insight,
            success_api=success_api,
            failure_api=failure_api,
            success_action=success_action,
            failure_action=failure_action,
        )
        return InsightRecord(
            task_id=str(ratio_record.get("task_id") or selected_sample.get("task_id") or ""),
            timestep=_optional_int(ratio_record.get("timestep")),
            state_summary=state_summary,
            success_action=success_action,
            failure_action=failure_action,
            delta_cf=delta_cf,
            insight=insight,
            ratio=_optional_float(ratio_record.get("ratio")),
            source="counterfactual_pair",
            selected_sample_id=str(selected_sample.get("sample_id", "")),
            evidence={
                "success_api": success_api,
                "failure_api": failure_api,
                "success_argument_keys": argument_keys,
                "delta_summary": delta_summary,
                "ratio": ratio_record.get("ratio"),
                "selected_bucket_reason": selected_sample.get("reason"),
            },
            validation_flags=validation_flags,
            rejected=bool(validation_flags),
        )


def extract_insight_records(
    *,
    selected_samples: list[dict[str, Any]],
    insight_extractor: InsightExtractor | None = None,
) -> list[InsightRecord]:
    extractor = insight_extractor or HeuristicInsightExtractor()
    return [
        extractor.extract_insight(selected_sample=selected_sample)
        for selected_sample in selected_samples
        if selected_sample.get("bucket") == "selected"
    ]


def summarize_state(state_snapshot: Any, *, max_length: int = 240) -> str:
    if not isinstance(state_snapshot, str) or not state_snapshot:
        return "No state snapshot recorded."
    try:
        messages = json.loads(state_snapshot)
    except json.JSONDecodeError:
        return _truncate(state_snapshot, max_length=max_length)
    if not isinstance(messages, list):
        return _truncate(state_snapshot, max_length=max_length)
    useful_contents: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role", "unknown")
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        useful_contents.append(f"{role}: {content}")
    if not useful_contents:
        return "No state messages recorded."
    return _truncate(" | ".join(useful_contents[-3:]), max_length=max_length)


def _build_insight_text(
    *,
    state_summary: str,
    success_api: str | None,
    argument_keys: list[str],
    failure_api: str | None,
    delta_summary: str,
) -> str:
    success_api_text = success_api or "the successful API"
    failure_api_text = failure_api or "the failed API"
    argument_text = ", ".join(argument_keys) if argument_keys else "no recorded keyword arguments"
    return (
        f"When the state is: {state_summary}, prefer API/action {success_api_text} "
        f"with argument pattern [{argument_text}], because failure action "
        f"{failure_api_text} differs in the counterfactual evidence: {delta_summary}"
    )


def _validation_flags(
    *,
    insight: str,
    success_api: str | None,
    failure_api: str | None,
    success_action: str | None,
    failure_action: str | None,
) -> list[str]:
    flags: list[str] = []
    if success_action is None:
        flags.append("missing_success_action")
    if failure_action is None:
        flags.append("missing_failure_action")
    if success_api is None:
        flags.append("missing_success_api")
    elif success_api not in insight:
        flags.append("insight_missing_success_api")
    if failure_api is None:
        flags.append("missing_failure_api")
    elif failure_api not in insight:
        flags.append("insight_missing_failure_api")
    return flags


def _ratio_record(selected_sample: dict[str, Any]) -> dict[str, Any]:
    ratio_record = selected_sample.get("ratio_record")
    return ratio_record if isinstance(ratio_record, dict) else {}


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _api_name(signature: dict[str, Any], action: str | None) -> str | None:
    full_api_name = signature.get("full_api_name")
    if isinstance(full_api_name, str) and full_api_name:
        return full_api_name
    app_name = signature.get("app_name")
    api_name = signature.get("api_name")
    if app_name and api_name:
        return f"{app_name}.{api_name}"
    if action is None:
        return None
    marker = "apis."
    if marker not in action:
        return None
    call_prefix = action.split(marker, 1)[1].split("(", 1)[0]
    parts = call_prefix.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return None


def _argument_keys(signature: dict[str, Any]) -> list[str]:
    argument_keys = signature.get("argument_keys")
    if not isinstance(argument_keys, list):
        return []
    return [str(argument_key) for argument_key in argument_keys]


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


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


def _truncate(value: str, *, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    return value[: max_length - 3] + "..."
