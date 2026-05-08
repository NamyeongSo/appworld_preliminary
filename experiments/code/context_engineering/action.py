from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActionSignature:
    """Normalized AppWorld API-call action used by context-engineering phases."""

    app_name: str
    api_name: str
    argument_keys: tuple[str, ...]

    @property
    def full_api_name(self) -> str:
        return f"{self.app_name}.{self.api_name}"

    def match_key(self, action_match_level: str = "api_name_plus_argument_keys") -> str:
        if action_match_level == "api_name":
            return self.full_api_name
        if action_match_level in {
            "api_name_plus_argument_keys",
            "api_name_plus_normalized_arguments",
        }:
            argument_keys = ",".join(self.argument_keys)
            return f"{self.full_api_name}({argument_keys})"
        raise ValueError(f"Unsupported action_match_level: {action_match_level!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "app_name": self.app_name,
            "api_name": self.api_name,
            "full_api_name": self.full_api_name,
            "argument_keys": list(self.argument_keys),
        }


@dataclass(frozen=True)
class ActionParseResult:
    signatures: tuple[ActionSignature, ...]
    parsing_error: str | None = None

    @property
    def primary_signature(self) -> ActionSignature | None:
        return self.signatures[0] if self.signatures else None

    def to_dict(self) -> dict[str, Any]:
        primary_signature = self.primary_signature
        return {
            "action_signature": (
                primary_signature.to_dict() if primary_signature is not None else None
            ),
            "action_signatures": [
                action_signature.to_dict() for action_signature in self.signatures
            ],
            "action_parse_error": self.parsing_error,
        }


@dataclass(frozen=True)
class StepRepresentation:
    state_snapshot: str
    action_signature: ActionSignature | None
    action_signatures: tuple[ActionSignature, ...]
    action_parse_error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_snapshot": self.state_snapshot,
            "action_signature": (
                self.action_signature.to_dict() if self.action_signature is not None else None
            ),
            "action_signatures": [
                action_signature.to_dict() for action_signature in self.action_signatures
            ],
            "action_parse_error": self.action_parse_error,
        }


def extract_action_signatures(action: str) -> ActionParseResult:
    """Extract normalized ``apis.<app>.<api>(...)`` signatures from ReAct code."""

    try:
        tree = ast.parse(action or "")
    except SyntaxError as exception:
        return ActionParseResult(
            signatures=(),
            parsing_error=(
                f"SyntaxError at line {exception.lineno}, "
                f"offset {exception.offset}: {exception.msg}"
            ),
        )

    visitor = _ActionSignatureVisitor()
    visitor.visit(tree)
    signatures = visitor.signatures

    if not signatures:
        return ActionParseResult(
            signatures=(),
            parsing_error="No AppWorld API call matching apis.<app>.<api>(...) found.",
        )
    return ActionParseResult(signatures=tuple(signatures))


def build_step_representation(
    *, messages_before_action: list[dict[str, Any]], action: str
) -> StepRepresentation:
    action_parse_result = extract_action_signatures(action)
    return StepRepresentation(
        state_snapshot=render_state_snapshot(messages_before_action),
        action_signature=action_parse_result.primary_signature,
        action_signatures=action_parse_result.signatures,
        action_parse_error=action_parse_result.parsing_error,
    )


def render_state_snapshot(messages_before_action: list[dict[str, Any]]) -> str:
    """Render the pre-action conversation state as deterministic JSON."""

    return json.dumps(
        messages_before_action,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def action_signatures_match(
    left: ActionSignature,
    right: ActionSignature,
    *,
    action_match_level: str = "api_name_plus_argument_keys",
) -> bool:
    """Return whether two normalized actions are equivalent at the requested level."""

    return left.match_key(action_match_level) == right.match_key(action_match_level)


def normalize_argument_keys(keywords: list[ast.keyword]) -> tuple[str, ...]:
    """Normalize keyword argument names so formatting/order differences disappear."""

    argument_keys = {keyword.arg if keyword.arg is not None else "**" for keyword in keywords}
    return tuple(sorted(argument_keys))


def _signature_from_call(node: ast.Call) -> ActionSignature | None:
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    api_name = func.attr
    app_attribute = func.value
    if not isinstance(app_attribute, ast.Attribute):
        return None
    app_name = app_attribute.attr
    root = app_attribute.value
    if not isinstance(root, ast.Name) or root.id != "apis":
        return None
    return ActionSignature(
        app_name=app_name,
        api_name=api_name,
        argument_keys=normalize_argument_keys(node.keywords),
    )


class _ActionSignatureVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.signatures: list[ActionSignature] = []

    def visit_Call(self, node: ast.Call) -> None:
        action_signature = _signature_from_call(node)
        if action_signature is not None:
            self.signatures.append(action_signature)
        self.generic_visit(node)
