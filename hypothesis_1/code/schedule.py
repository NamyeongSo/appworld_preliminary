from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hypothesis_1.code.config import (
    ALL_CONDITIONS,
    BASE_SEED,
    PASS10_SEEDS,
    REFLECTION_CONDITION_TO_SOURCE,
    VANILLA_CONDITIONS,
)
from hypothesis_1.code.paths import DEFAULT_REFLECTION_MANIFEST_PATH


@dataclass(frozen=True)
class RunSpec:
    condition: str
    task_id: str
    run_id: str
    seed: int
    sample_id: str | None = None
    reflection_source: str | None = None
    guidance_text: str | None = None
    manifest_index: int | None = None

    @property
    def experiment_name(self) -> str:
        return f"hypothesis_1/{self.condition}/{self.run_id}"

    @property
    def flat_run_id(self) -> str:
        return self.run_id.replace("/", "__")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["experiment_name"] = self.experiment_name
        return data


def load_manifest(path: Path = DEFAULT_REFLECTION_MANIFEST_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def unique_task_ids(manifest: dict[str, Any]) -> list[str]:
    if "unique_task_ids" in manifest:
        return sorted(manifest["unique_task_ids"])
    return sorted({sample["task_id"] for sample in manifest.get("samples", [])})


def build_run_specs(
    condition: str,
    manifest: dict[str, Any],
    *,
    limit: int | None = None,
    task_id: str | None = None,
    sample_id: str | None = None,
) -> list[RunSpec]:
    if condition not in ALL_CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {ALL_CONDITIONS}")

    specs: list[RunSpec] = []
    if condition in REFLECTION_CONDITION_TO_SOURCE:
        source_name = REFLECTION_CONDITION_TO_SOURCE[condition]
        for index, sample in enumerate(manifest.get("samples", [])):
            if task_id and sample["task_id"] != task_id:
                continue
            if sample_id and sample["sample_id"] != sample_id:
                continue
            reflection = sample["reflections"][source_name]
            specs.append(
                RunSpec(
                    condition=condition,
                    task_id=sample["task_id"],
                    run_id=f"{sample['sample_id']}__{source_name}",
                    seed=BASE_SEED,
                    sample_id=sample["sample_id"],
                    reflection_source=source_name,
                    guidance_text=reflection["guidance"],
                    manifest_index=index,
                )
            )
    elif condition in VANILLA_CONDITIONS:
        seeds = [BASE_SEED] if condition == "vanilla_react_pass1" else PASS10_SEEDS
        for task_id_ in unique_task_ids(manifest):
            if task_id and task_id_ != task_id:
                continue
            for seed in seeds:
                specs.append(
                    RunSpec(
                        condition=condition,
                        task_id=task_id_,
                        run_id=f"{task_id_}/seed_{seed}",
                        seed=seed,
                    )
                )

    if limit is not None:
        specs = specs[:limit]
    return specs

