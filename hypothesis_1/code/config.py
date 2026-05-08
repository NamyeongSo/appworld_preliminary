from __future__ import annotations

import copy
from typing import Any

from hypothesis_1.code.paths import REACT_PROMPT_PATH


BASE_SEED = 100
PASS10_SEEDS = [BASE_SEED + index for index in range(10)]
DEFAULT_DATASET = "test_normal"

REFLECTION_CONDITION_TO_SOURCE = {
    "reflection_analysis_pass1": "analysis",
    "reflection_random_pairs_pass1": "analysis_random_pairs",
    "reflection_random_3_pass1": "analysis_random_3",
    "reflection_random_4_pass1": "analysis_random_4",
    "reflection_random_5_pass1": "analysis_random_5",
    "reflection_random_6_pass1": "analysis_random_6",
    "reflection_random_7_pass1": "analysis_random_7",
    "reflection_root_pass1": "analysis_root",
}
VANILLA_CONDITIONS = {"vanilla_react_pass1", "vanilla_react_pass10"}
ALL_CONDITIONS = tuple(
    [
        *REFLECTION_CONDITION_TO_SOURCE.keys(),
        "vanilla_react_pass1",
        "vanilla_react_pass10",
    ]
)

DEFAULT_MODEL_CONFIG: dict[str, Any] = {
    "client_name": "openai",
    "api_type": "chat_completions",
    "base_url": "{MODEL_SERVER_URL}/v1",
    "api_key_env_name": "NO_API_KEY",
    "name": "google/gemma-4-26B-A4B-it",
    "temperature": 0.5,
    "seed": BASE_SEED,
    "extra_body": {
        "chat_template_kwargs": {"enable_thinking": True},
        "skip_special_tokens": False,
    },
    "drop_reasoning_content": False,
    "cost_per_token": {
        "input_cache_hit": 0.0,
        "input_cache_miss": 0.0,
        "input_cache_write": 0.0,
        "output": 0.0,
    },
    "retry_after_n_seconds": 15,
    "use_cache": False,
    "max_retries": 100,
}

DEFAULT_AGENT_CONFIG: dict[str, Any] = {
    "type": "simplified_react_code_agent",
    "model_config": DEFAULT_MODEL_CONFIG,
    "appworld_config": {
        "random_seed": BASE_SEED,
        "raise_on_extra_parameters": True,
    },
    "logger_config": {
        "color": True,
        "verbose": True,
    },
    "usage_tracker_config": {
        "max_cost_overall": 1000,
        "max_cost_per_task": 10,
        "max_output_tokens_per_task": 100000,
    },
    "prompt_file_path": str(REACT_PROMPT_PATH),
    "ignore_multiple_calls": True,
    "max_prompt_length": None,
    "max_output_length": None,
    "max_steps": 50,
    "log_lm_calls": True,
    "skip_if_finished": True,
}


def build_agent_config(
    *,
    seed: int,
    verbose: bool,
    color: bool,
    log_lm_calls: bool,
    guidance_text: str | None = None,
    guidance_source: str | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_AGENT_CONFIG)
    config["model_config"]["seed"] = seed
    config["appworld_config"]["random_seed"] = seed
    config["logger_config"]["verbose"] = verbose
    config["logger_config"]["color"] = color
    config["log_lm_calls"] = log_lm_calls
    if guidance_text is not None:
        config["type"] = "hypothesis_1_reflection_react_code_agent"
        config["guidance_text"] = guidance_text
        config["guidance_source"] = guidance_source
    return config
