from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hypothesis_1.code.config import REFLECTION_CONDITION_TO_SOURCE
from hypothesis_1.code.guidance import extract_guidance_text
from hypothesis_1.code.paths import (
    DEFAULT_COUNTERFACTUAL_OUTPUTS_ROOT,
    DEFAULT_REFLECTION_MANIFEST_PATH,
    ensure_hypothesis_tree,
    path_to_jsonable,
    resolve_repo_path,
)


REFLECTION_SOURCES = tuple(REFLECTION_CONDITION_TO_SOURCE.values())
SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def make_sample_id(outputs_root: Path, counterfactual_dir: Path) -> str:
    relative = counterfactual_dir.relative_to(outputs_root)
    parts = list(relative.parts)
    if parts and parts[-1] == "counterfactual":
        parts = parts[:-1]
    if "tasks" in parts:
        parts.remove("tasks")
    sample_id = "__".join(parts)
    return SAFE_ID_RE.sub("_", sample_id).strip("_")


def _read_response_text(json_path: Path, payload: dict[str, Any]) -> str:
    response_text = payload.get("response_text")
    if isinstance(response_text, str) and response_text.strip():
        return response_text

    response_path = payload.get("response_path")
    candidate_paths: list[Path] = []
    if isinstance(response_path, str) and response_path:
        candidate_paths.append(resolve_repo_path(response_path))
    candidate_paths.append(json_path.with_suffix(".txt"))
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path.read_text(encoding="utf-8")
    return ""


def load_reflection(reflection_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    candidates = sorted(reflection_dir.glob("*.json"))
    if not candidates:
        raise ValueError(f"No reflection JSON files found in {reflection_dir}")
    for json_path in candidates:
        try:
            payload = read_json(json_path)
            response_text = _read_response_text(json_path, payload)
            try:
                guidance = extract_guidance_text(response_text)
                has_guidance_tag = True
            except ValueError:
                guidance = ""
                has_guidance_tag = False
            return {
                "json_path": path_to_jsonable(json_path),
                "txt_path": path_to_jsonable(json_path.with_suffix(".txt"))
                if json_path.with_suffix(".txt").exists()
                else None,
                "guidance": guidance,
                "guidance_chars": len(guidance),
                "has_guidance_tag": has_guidance_tag,
                "candidate_count": len(candidates),
            }
        except Exception as exception:  # Try the next artifact in deterministic order.
            errors.append(f"{json_path.name}: {type(exception).__name__}: {exception}")
    raise ValueError(
        f"No usable reflection guidance found in {reflection_dir}. Errors: {'; '.join(errors[:5])}"
    )


def build_reflection_manifest(
    *,
    outputs_root: Path = DEFAULT_COUNTERFACTUAL_OUTPUTS_ROOT,
    require_pair_complete: bool = False,
) -> dict[str, Any]:
    outputs_root = outputs_root.resolve()
    samples: list[dict[str, Any]] = []
    skipped: dict[str, int] = {
        "pair_incomplete": 0,
        "missing_reflection_source": 0,
    }
    empty_guidance = 0

    for manifest_path in sorted(outputs_root.glob("**/counterfactual/manifest.json")):
        counterfactual_dir = manifest_path.parent
        manifest = read_json(manifest_path)
        if require_pair_complete and not bool(manifest.get("pair_complete", False)):
            skipped["pair_incomplete"] += 1
            continue

        reflections: dict[str, Any] = {}
        missing_source = False
        for source_name in REFLECTION_SOURCES:
            reflection_dir = counterfactual_dir / source_name
            if not reflection_dir.is_dir():
                missing_source = True
                break
            try:
                reflections[source_name] = load_reflection(reflection_dir)
            except ValueError:
                missing_source = True
                break
        if missing_source:
            skipped["missing_reflection_source"] += 1
            continue
        empty_guidance += sum(
            1 for reflection in reflections.values() if not reflection["guidance"]
        )

        sample_id = make_sample_id(outputs_root, counterfactual_dir)
        samples.append(
            {
                "sample_id": sample_id,
                "task_id": manifest["task_id"],
                "case": manifest.get("case"),
                "root_success": manifest.get("root_success"),
                "pair_complete": manifest.get("pair_complete"),
                "final_success_trajectory_id": manifest.get("final_success_trajectory_id"),
                "final_failure_trajectory_id": manifest.get("final_failure_trajectory_id"),
                "num_trajectories": manifest.get("num_trajectories"),
                "manifest_path": path_to_jsonable(manifest_path),
                "counterfactual_dir": path_to_jsonable(counterfactual_dir),
                "reflections": reflections,
            }
        )

    sample_ids = [sample["sample_id"] for sample in samples]
    duplicate_sample_ids = sorted(
        sample_id for sample_id in set(sample_ids) if sample_ids.count(sample_id) > 1
    )
    if duplicate_sample_ids:
        raise ValueError(f"Duplicate sample_ids generated: {duplicate_sample_ids[:10]}")

    unique_tasks = sorted({sample["task_id"] for sample in samples})
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "outputs_root": path_to_jsonable(outputs_root),
        "require_pair_complete": require_pair_complete,
        "reflection_sources": list(REFLECTION_SOURCES),
        "num_samples": len(samples),
        "num_unique_tasks": len(unique_tasks),
        "unique_task_ids": unique_tasks,
        "skipped": skipped,
        "empty_guidance_artifacts": empty_guidance,
        "samples": samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the hypothesis 1 reflection manifest.")
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=DEFAULT_COUNTERFACTUAL_OUTPUTS_ROOT,
        help="Counterfactual outputs root to scan.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REFLECTION_MANIFEST_PATH,
        help="Manifest path to write.",
    )
    parser.add_argument(
        "--require-pair-complete",
        action="store_true",
        help="Only include counterfactual manifests where pair_complete is true.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_hypothesis_tree()
    manifest = build_reflection_manifest(
        outputs_root=args.outputs_root,
        require_pair_complete=args.require_pair_complete,
    )
    write_json(manifest, args.output)
    print(
        "Wrote "
        f"{manifest['num_samples']} samples across {manifest['num_unique_tasks']} tasks "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
