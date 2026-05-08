from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HYPOTHESIS_ROOT = REPO_ROOT / "hypothesis_1"
CODE_DIR = HYPOTHESIS_ROOT / "code"
DATA_DIR = HYPOTHESIS_ROOT / "data"
PROMPTS_DIR = HYPOTHESIS_ROOT / "prompts"
OUTPUTS_DIR = HYPOTHESIS_ROOT / "outputs"
REPORTS_DIR = HYPOTHESIS_ROOT / "reports"
APPWORLD_ROOT = HYPOTHESIS_ROOT / "appworld_root"
APPWORLD_OUTPUTS_DIR = APPWORLD_ROOT / "experiments" / "outputs"

DEFAULT_COUNTERFACTUAL_OUTPUTS_ROOT = (
    REPO_ROOT / "experiments" / "outputs" / "counterfactual" / "google" / "gemma-4-26b-a4b-it"
)
DEFAULT_REFLECTION_MANIFEST_PATH = DATA_DIR / "reflection_manifest.json"
DEFAULT_REPORT_PATH = REPORTS_DIR / "hypothesis_1_report.md"
DEFAULT_METRICS_PATH = OUTPUTS_DIR / "hypothesis_1_metrics.json"
REACT_PROMPT_PATH = REPO_ROOT / "experiments" / "prompts" / "react_code_agent" / "instructions.txt"


def ensure_hypothesis_tree() -> None:
    for path in (
        CODE_DIR,
        DATA_DIR,
        PROMPTS_DIR,
        OUTPUTS_DIR,
        REPORTS_DIR,
        APPWORLD_OUTPUTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_appworld_root() -> None:
    ensure_hypothesis_tree()
    data_link = APPWORLD_ROOT / "data"
    source_data = REPO_ROOT / "data"
    if not data_link.exists():
        data_link.symlink_to(source_data, target_is_directory=True)


def configure_appworld_root() -> None:
    ensure_appworld_root()
    os.environ["APPWORLD_ROOT"] = str(APPWORLD_ROOT)
    os.environ.setdefault("MODEL_SERVER_URL", "http://localhost:8002")


def path_to_jsonable(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path

