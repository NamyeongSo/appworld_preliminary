from __future__ import annotations

import json
from typing import Any

from jinja2 import Template

from appworld import AppWorld
from appworld_agents.code.simplified.agent import Agent
from appworld_agents.code.simplified.react_code_agent import SimplifiedReActCodeAgent

from hypothesis_1.code.guidance import inject_guidance


@Agent.register("hypothesis_1_reflection_react_code_agent")
class ReflectionGuidedReActCodeAgent(SimplifiedReActCodeAgent):  # type: ignore[misc]
    def __init__(
        self,
        guidance_text: str,
        guidance_source: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.guidance_text = guidance_text
        self.guidance_source = guidance_source

    def initialize(self, world: AppWorld) -> None:
        Agent.initialize(self, world)
        template = Template(self.prompt_template)
        app_descriptions = json.dumps(
            [{"name": key, "description": value} for key, value in world.task.app_descriptions.items()],
            indent=1,
        )
        instruction = inject_guidance(
            world.task.instruction,
            self.guidance_text,
            source_name=self.guidance_source,
        )
        output_str = template.render(
            {
                "instruction": instruction,
                "main_user": world.task.supervisor,
                "app_descriptions": app_descriptions,
            }
        )
        output_str = self.truncate_input(output_str) + "\n\n"
        self.messages = self.text_to_messages(output_str)
        self.num_instruction_messages = len(self.messages)

