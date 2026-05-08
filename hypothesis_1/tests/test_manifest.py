import json
from pathlib import Path

from hypothesis_1.code.config import REFLECTION_CONDITION_TO_SOURCE
from hypothesis_1.code.manifest import build_reflection_manifest, make_sample_id


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_build_reflection_manifest_requires_all_sources(tmp_path: Path) -> None:
    counterfactual_dir = (
        tmp_path / "chunk_001" / "run_01" / "tasks" / "abc_1" / "counterfactual"
    )
    write_json(
        counterfactual_dir / "manifest.json",
        {
            "task_id": "abc_1",
            "case": "A",
            "pair_complete": True,
            "root_success": False,
            "final_success_trajectory_id": "case_a",
            "final_failure_trajectory_id": "root",
            "num_trajectories": 2,
        },
    )
    for source in REFLECTION_CONDITION_TO_SOURCE.values():
        write_json(
            counterfactual_dir / source / "reflection.json",
            {"response_text": f"<guidance>{source} guidance</guidance>"},
        )

    manifest = build_reflection_manifest(outputs_root=tmp_path)

    assert manifest["num_samples"] == 1
    assert manifest["num_unique_tasks"] == 1
    sample = manifest["samples"][0]
    assert sample["task_id"] == "abc_1"
    assert sample["reflections"]["analysis"]["guidance"] == "analysis guidance"
    assert sample["reflections"]["analysis_random_pairs"]["guidance"] == (
        "analysis_random_pairs guidance"
    )


def test_build_reflection_manifest_skips_missing_source(tmp_path: Path) -> None:
    counterfactual_dir = tmp_path / "chunk" / "run" / "tasks" / "abc_1" / "counterfactual"
    write_json(counterfactual_dir / "manifest.json", {"task_id": "abc_1", "pair_complete": True})
    write_json(
        counterfactual_dir / "analysis" / "reflection.json",
        {"response_text": "<guidance>ok</guidance>"},
    )

    manifest = build_reflection_manifest(outputs_root=tmp_path)

    assert manifest["num_samples"] == 0
    assert manifest["skipped"]["missing_reflection_source"] == 1


def test_make_sample_id_is_stable(tmp_path: Path) -> None:
    counterfactual_dir = tmp_path / "chunk" / "run_01" / "tasks" / "abc_1" / "counterfactual"
    assert make_sample_id(tmp_path, counterfactual_dir) == "chunk__run_01__abc_1"
