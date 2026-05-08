local experiment_prompts_path = std.extVar("APPWORLD_EXPERIMENT_PROMPTS_PATH");
local experiment_configs_path = std.extVar("APPWORLD_EXPERIMENT_CONFIGS_PATH");
local experiment_code_path = std.extVar("APPWORLD_EXPERIMENT_CODE_PATH");
{
    "type": "counterfactual",
    "config": {
        "model_server": {
            "command": "vllm serve google/gemma-4-26B-A4B-it --max-num-seqs 5 --max-model-len 32768 --gpu-memory-utilization 0.90 --limit-mm-per-prompt image=0,audio=0 --enable-auto-tool-choice --reasoning-parser gemma4 --tool-call-parser gemma4 --chat-template examples/tool_chat_template_gemma4.jinja --port {port}",
            "enabled": true,
            "health_check_at": "/v1/models",
            "port": 8002,
            "show_logs": false,
            "started": true,
            "timeout": 600
        },
        "agent": {
            "type": "simplified_react_code_agent",
            "model_config": {
                "client_name": "openai",
                "api_type": "chat_completions",
                "base_url": "{MODEL_SERVER_URL}/v1",
                "api_key_env_name": "NO_API_KEY",
                "name": "google/gemma-4-26B-A4B-it",
                "temperature": 0.5,
                "seed": 128,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": true}, "skip_special_tokens": false},
                "drop_reasoning_content": false,
                "cost_per_token": {"input_cache_hit": 0.0, "input_cache_miss": 0.0, "input_cache_write": 0.0, "output": 0.0},
                "retry_after_n_seconds": 15,
                "use_cache": false,
                "max_retries": 100,
            },
            "appworld_config": {
                "random_seed": 128,
                "raise_on_extra_parameters": true,
            },
            "logger_config": {
                "color": true,
                "verbose": true,
            },
            "usage_tracker_config": {
                "max_cost_overall": 1000,
                "max_cost_per_task": 10,
                "max_output_tokens_per_task": 100000,
            },
            "prompt_file_path": experiment_prompts_path + "/react_code_agent/instructions.txt",
            "ignore_multiple_calls": true,
            "max_prompt_length": null,
            "max_output_length": null,
            "max_steps": 50,
            "log_lm_calls": true,
            "skip_if_finished": true,
        },
        "dataset": "test_normal",
        "counterfactual": {
            "max_replay_attempts": 50,
            "case_b_max_attempts": 50,
            "copy_attempt_dbs": true,
            "ensure_different_action": true,
            "random_seed": 128,
        },
    },
    "metadata": {
        "model": {
            "file_name": "gemma-4-26b-a4b-it",
            "humanized_name": "Gemma 4 26B A4B It",
            "precise_name": "google/gemma-4-26B-A4B-it",
            "creator": "google",
            "provider": "vllm",
        },
        "agent": {
            "file_name": "simplified_react_code_agent",
            "humanized_name": "ReAct Code Agent",
        },
    },
}
