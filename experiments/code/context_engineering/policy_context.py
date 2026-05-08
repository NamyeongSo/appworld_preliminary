from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from appworld_agents.code.context_engineering.insight import InsightRecord


@dataclass(frozen=True)
class PolicyContextItem:
    """Prompt-ready policy context item derived from a validated insight."""

    context_id: str
    task_id: str
    timestep: int | None
    insight: str
    state_summary: str
    success_action: str | None
    failure_action: str | None
    delta_cf: dict[str, Any]
    ratio: float | None
    source: str
    selected_sample_id: str
    evidence: dict[str, Any]
    created_loop_index: int
    created_at_utc: str

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> PolicyContextItem:
        return cls(
            context_id=str(data["context_id"]),
            task_id=str(data["task_id"]),
            timestep=_optional_int(data.get("timestep")),
            insight=str(data["insight"]),
            state_summary=str(data.get("state_summary", "")),
            success_action=_optional_str(data.get("success_action")),
            failure_action=_optional_str(data.get("failure_action")),
            delta_cf=_dict_value(data.get("delta_cf")),
            ratio=_optional_float(data.get("ratio")),
            source=str(data.get("source", "")),
            selected_sample_id=str(data.get("selected_sample_id", "")),
            evidence=_dict_value(data.get("evidence")),
            created_loop_index=int(data["created_loop_index"]),
            created_at_utc=str(data["created_at_utc"]),
        )

    @classmethod
    def from_insight_record(
        cls,
        *,
        insight_record: InsightRecord,
        created_loop_index: int,
        created_at_utc: str,
    ) -> PolicyContextItem:
        context_id = make_policy_context_id(
            task_id=insight_record.task_id,
            timestep=insight_record.timestep,
            insight=insight_record.insight,
        )
        return cls(
            context_id=context_id,
            task_id=insight_record.task_id,
            timestep=insight_record.timestep,
            insight=insight_record.insight,
            state_summary=insight_record.state_summary,
            success_action=insight_record.success_action,
            failure_action=insight_record.failure_action,
            delta_cf=insight_record.delta_cf,
            ratio=insight_record.ratio,
            source=insight_record.source,
            selected_sample_id=insight_record.selected_sample_id,
            evidence=insight_record.evidence,
            created_loop_index=created_loop_index,
            created_at_utc=created_at_utc,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "task_id": self.task_id,
            "timestep": self.timestep,
            "insight": self.insight,
            "state_summary": self.state_summary,
            "success_action": self.success_action,
            "failure_action": self.failure_action,
            "delta_cf": self.delta_cf,
            "ratio": self.ratio,
            "source": self.source,
            "selected_sample_id": self.selected_sample_id,
            "evidence": self.evidence,
            "created_loop_index": self.created_loop_index,
            "created_at_utc": self.created_at_utc,
        }


def build_policy_context_items(
    *,
    insight_records: Iterable[InsightRecord],
    created_loop_index: int,
    created_at_utc: str,
    include_rejected: bool = False,
) -> list[PolicyContextItem]:
    return [
        PolicyContextItem.from_insight_record(
            insight_record=insight_record,
            created_loop_index=created_loop_index,
            created_at_utc=created_at_utc,
        )
        for insight_record in insight_records
        if include_rejected or not insight_record.rejected
    ]


def merge_policy_context_items(
    *,
    existing_items: list[PolicyContextItem],
    new_items: list[PolicyContextItem],
    max_context_items: int,
) -> list[PolicyContextItem]:
    """Merge by stable context id, preserving order and trimming to latest items."""

    context_id_to_item: dict[str, PolicyContextItem] = {}
    ordered_items: list[PolicyContextItem] = []
    for item in [*existing_items, *new_items]:
        if item.context_id in context_id_to_item:
            continue
        context_id_to_item[item.context_id] = item
        ordered_items.append(item)
    if len(ordered_items) <= max_context_items:
        return ordered_items
    return ordered_items[-max_context_items:]


def render_policy_context_prompt(
    *,
    policy_context_items: list[PolicyContextItem],
    all_insight: bool,
    max_context_items: int,
) -> str:
    selected_items = (
        policy_context_items if all_insight else policy_context_items[-max_context_items:]
    )
    lines = [
        "# Policy Context",
        "",
        "Use these learned counterfactual lessons as additional decision guidance. "
        "They do not override the task instructions or API documentation.",
        "",
        f"- num_context_items: {len(selected_items)}",
        f"- all_insight: {str(all_insight).lower()}",
        f"- max_context_items: {max_context_items}",
        "",
        "## Insights",
    ]
    if not selected_items:
        lines.extend(["", "No policy insights are available yet."])
    for item_index, item in enumerate(selected_items, start=1):
        lines.extend(
            [
                "",
                f"{item_index}. context_id: `{item.context_id}`",
                f"   - task_id: `{item.task_id}`",
                f"   - timestep: {item.timestep}",
                f"   - ratio: {item.ratio}",
                f"   - insight: {item.insight}",
            ]
        )
        if item.success_action:
            lines.append(f"   - success_action: `{item.success_action}`")
        if item.failure_action:
            lines.append(f"   - failure_action: `{item.failure_action}`")
    prompt_without_token_count = "\n".join(lines).rstrip() + "\n"
    token_count = estimate_token_count(prompt_without_token_count)
    lines.insert(6, f"- estimated_token_count: {token_count}")
    return "\n".join(lines).rstrip() + "\n"


def inject_policy_context_prompt_template(
    *,
    original_prompt_template: str,
    context_prompt: str,
) -> str:
    context_message = (
        "USER:\n"
        "Additional policy context learned from prior counterfactuals:\n\n"
        f"{context_prompt.rstrip()}\n\n"
    )
    final_task_marker = "USER:\nUsing these APIs, now generate code to solve the actual task:"
    insertion_index = original_prompt_template.rfind(final_task_marker)
    if insertion_index != -1:
        return (
            original_prompt_template[:insertion_index].rstrip()
            + "\n\n"
            + context_message
            + original_prompt_template[insertion_index:]
        )
    return original_prompt_template.rstrip() + "\n\n" + context_message


def make_policy_context_id(*, task_id: str, timestep: int | None, insight: str) -> str:
    payload = json.dumps(
        {"task_id": task_id, "timestep": timestep, "insight": insight},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def estimate_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _optional_str(value: Any) -> str | None:
    return str(value) if value is not None else None


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
