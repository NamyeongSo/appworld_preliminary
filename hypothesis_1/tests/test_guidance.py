from hypothesis_1.code.guidance import extract_guidance_text, inject_guidance


def test_extract_guidance_text_uses_final_block() -> None:
    text = """
    <guidance>first</guidance>
    analysis...
    <guidance>
    second
    </guidance>
    """
    assert extract_guidance_text(text) == "second"


def test_inject_guidance_appends_tagged_strategy() -> None:
    prompt = inject_guidance("Do the task.", "Check all pages.", source_name="analysis")
    assert prompt.startswith("Do the task.")
    assert "Task-specific guidance" in prompt
    assert "<guidance>\nCheck all pages.\n</guidance>" in prompt
    assert "analysis" in prompt

