from __future__ import annotations

import re


GUIDANCE_RE = re.compile(r"<guidance>\s*(.*?)\s*</guidance>", flags=re.IGNORECASE | re.DOTALL)


def extract_guidance_text(response_text: str, *, require_tags: bool = True) -> str:
    """Extract the final guidance block from a reflection response."""
    matches = GUIDANCE_RE.findall(response_text or "")
    if matches:
        return matches[-1].strip()
    if require_tags:
        raise ValueError("No <guidance>...</guidance> block found.")
    return (response_text or "").strip()


def guidance_block(guidance_text: str) -> str:
    guidance_text = guidance_text.strip()
    if not guidance_text:
        return ""
    return f"<guidance>\n{guidance_text}\n</guidance>"


def inject_guidance(
    instruction: str,
    guidance_text: str,
    *,
    source_name: str | None = None,
) -> str:
    """Append reflection guidance to the task instruction seen by the ReAct agent."""
    guidance_text = guidance_text.strip()
    if not guidance_text:
        return instruction
    source_suffix = f" ({source_name})" if source_name else ""
    return (
        instruction.rstrip()
        + "\n\nTask-specific guidance from prior counterfactual reflection"
        + source_suffix
        + ":\n"
        + guidance_block(guidance_text)
        + "\n\nUse this guidance as strategy advice. Still inspect the current API docs and app state before acting."
    )

