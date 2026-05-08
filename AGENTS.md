# Repository Guidelines

## Project Structure & Module Organization
- `appworld/` contains the main AppWorld package after running `appworld install --repo`; keep core runtime, CLI, and verification changes there.
- `experiments/code/` holds agent implementations (`simplified/`, `smolagents/`, `openai_agents/`, `legacy/`) and shared utilities in `common/`.
- `experiments/configs/` stores runnable experiment configs; `experiments/prompts/` stores prompt templates.
- `data/` contains benchmark assets (`api_docs/`, `base_dbs/`, `datasets/`, `tasks/`). Treat large data changes carefully and avoid incidental churn.
- `generate/` is for data-generation scripts; `.github/workflows/` mirrors CI expectations.

## Build, Test, and Development Commands
- `uv venv && source .venv/bin/activate` creates the recommended local environment.
- `uv build --wheel` builds the package exactly as CI does.
- `uv pip install -e experiments` installs the companion `appworld-agents` package for local agent work.
- `uv run --no-editable appworld install --repo` unpacks the main package into this checkout.
- `uv run --no-editable appworld verify tests --root appworld-root` runs test verification.
- `uv run --no-editable appworld verify tasks --root appworld-root --num-processes 2 --include-only-first-n-tasks 4` is the fastest representative task-level smoke test.

## Coding Style & Naming Conventions
- Target Python 3.11+ with full type annotations.
- Format with `ruff format .`, lint with `ruff check .`, and type-check with `mypy --no-warn-unused-ignores .`.
- Use `lowercase_with_underscores` for functions/variables, `CamelCase` for classes, and `UPPERCASE_WITH_UNDERSCORES` for constants.
- Follow `DEVELOPMENT.md`: pluralize collection names (`task_ids`), use `key_to_value` for mappings (`user_id_to_email`), and avoid opaque abbreviations.

## Testing Guidelines
- Install hooks with `pre-commit install`; the repo runs Ruff checks before commits.
- Prefer focused verification first, then broaden to `appworld verify tasks` when behavior touches execution, APIs, or datasets.
- Keep test-related changes deterministic; CI fixes `PYTHONHASHSEED=0` and uses small task subsets on smoke runs.

## Commit & Pull Request Guidelines
- Recent history uses short, imperative subjects, often with issue/PR references, e.g. `Fix bundle files (#210)`.
- Follow the repository’s Lore commit protocol: explain why, then add trailers such as `Constraint:`, `Confidence:`, and `Tested:`.
- PRs should summarize user-visible impact, list verification commands run, link related issues, and call out any data, config, or prompt changes. Include screenshots only when docs or visual outputs change.
