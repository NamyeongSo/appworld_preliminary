from hypothesis_1.code.aggregate import (
    build_metrics,
    build_pass10_task_subset_metrics,
    records_normalized_to_five_runs,
    wilson_interval,
)
from hypothesis_1.code.config import ALL_CONDITIONS, REFLECTION_CONDITION_TO_SOURCE
from hypothesis_1.code.runner import build_selected_specs, selected_conditions
from hypothesis_1.code.schedule import build_run_specs


def manifest_fixture() -> dict:
    return {
        "samples": [
            {
                "sample_id": "sample_a",
                "task_id": "task_1",
                "reflections": {
                    source: {"guidance": f"sample_a {source}"}
                    for source in REFLECTION_CONDITION_TO_SOURCE.values()
                },
            },
            {
                "sample_id": "sample_b",
                "task_id": "task_2",
                "reflections": {
                    source: {"guidance": f"sample_b {source}"}
                    for source in REFLECTION_CONDITION_TO_SOURCE.values()
                },
            },
        ],
        "unique_task_ids": ["task_1", "task_2"],
    }


def test_build_run_specs_for_reflection_and_pass10() -> None:
    manifest = manifest_fixture()
    reflection_specs = build_run_specs("reflection_analysis_pass1", manifest)
    pass10_specs = build_run_specs("vanilla_react_pass10", manifest)

    assert len(reflection_specs) == 2
    assert reflection_specs[0].guidance_text == "sample_a analysis"
    assert len(pass10_specs) == 20
    assert {spec.task_id for spec in pass10_specs} == {"task_1", "task_2"}


def test_build_selected_specs_supports_all_conditions_and_run_id_filter() -> None:
    manifest = manifest_fixture()
    specs = build_selected_specs(condition="all", manifest=manifest)

    assert selected_conditions("all") == list(ALL_CONDITIONS)
    assert len(specs) == 2 * len(REFLECTION_CONDITION_TO_SOURCE) + 22
    assert [
        spec.condition
        for spec in build_selected_specs(
            condition="all",
            manifest=manifest,
            run_id="task_1/seed_100",
        )
    ] == ["vanilla_react_pass1", "vanilla_react_pass10"]


def test_build_selected_specs_run_id_filter_with_condition_is_exact() -> None:
    manifest = manifest_fixture()
    specs = build_selected_specs(
        condition="vanilla_react_pass10",
        manifest=manifest,
        run_id="task_1/seed_100",
    )

    assert len(specs) == 1
    assert specs[0].condition == "vanilla_react_pass10"


def test_build_metrics_computes_condition_and_pass10() -> None:
    records = [
        {"condition": "reflection_analysis_pass1", "task_id": "task_1", "success": False},
        {"condition": "reflection_analysis_pass1", "task_id": "task_2", "success": True},
        {"condition": "vanilla_react_pass1", "task_id": "task_1", "success": True},
        {"condition": "vanilla_react_pass1", "task_id": "task_2", "success": False},
        {"condition": "vanilla_react_pass10", "task_id": "task_1", "success": False},
        {"condition": "vanilla_react_pass10", "task_id": "task_1", "success": True},
        {"condition": "vanilla_react_pass10", "task_id": "task_2", "success": False},
    ]

    metrics = build_metrics(records=records)

    assert metrics["conditions"]["reflection_analysis_pass1"]["success_rate"] == 0.5
    assert metrics["vanilla_react_pass10"]["pass_at_10"] == 0.5
    assert metrics["paired_task_deltas"]["reflection_analysis_pass1"]["rows"][0][
        "delta_vs_vanilla_pass1"
    ] == -1.0


def test_records_normalized_to_five_runs_caps_pass10_and_repeats_pass1() -> None:
    records = [
        {
            "condition": "vanilla_react_pass1",
            "task_id": "task_1",
            "run_id": "task_1/seed_100",
            "seed": 100,
            "success": True,
        },
        *[
            {
                "condition": "vanilla_react_pass10",
                "task_id": "task_1",
                "run_id": f"task_1/seed_{100 + seed_index}",
                "seed": 100 + seed_index,
                "success": seed_index < 3,
            }
            for seed_index in range(10)
        ],
    ]

    normalized = records_normalized_to_five_runs(records)
    by_condition = {}
    for record in normalized:
        by_condition.setdefault(record["condition"], []).append(record)

    assert len(by_condition["vanilla_react_pass1"]) == 5
    assert len(by_condition["vanilla_react_pass10"]) == 5
    assert [record["seed"] for record in by_condition["vanilla_react_pass10"]] == [
        100,
        101,
        102,
        103,
        104,
    ]


def test_build_pass10_task_subset_metrics_restricts_all_conditions(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        __import__("json").dumps(manifest_fixture()),
        encoding="utf-8",
    )
    records = [
        {"condition": "reflection_analysis_pass1", "task_id": "task_1", "run_id": "sample_a__analysis", "success": True},
        {"condition": "reflection_analysis_pass1", "task_id": "task_2", "run_id": "sample_b__analysis", "success": False},
        {"condition": "vanilla_react_pass1", "task_id": "task_1", "run_id": "task_1/seed_100", "success": True},
        {"condition": "vanilla_react_pass1", "task_id": "task_2", "run_id": "task_2/seed_100", "success": True},
        {"condition": "vanilla_react_pass10", "task_id": "task_1", "run_id": "task_1/seed_100", "success": True},
    ]

    metrics = build_pass10_task_subset_metrics(records=records, manifest_path=manifest_path)

    assert metrics["subset"]["included_task_ids"] == ["task_1"]
    assert metrics["subset"]["excluded_task_ids"] == ["task_2"]
    assert metrics["conditions"]["reflection_analysis_pass1"]["num_tasks"] == 1
    assert metrics["conditions"]["reflection_analysis_pass1"]["success_rate"] == 1.0


def test_wilson_interval_empty_is_zero() -> None:
    assert wilson_interval(0, 0) == (0.0, 0.0)
