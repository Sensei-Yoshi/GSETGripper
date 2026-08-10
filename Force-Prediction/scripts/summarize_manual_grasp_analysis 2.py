"""Export a verified table of original and manual-contact Marigold results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

MANUAL_STRATEGY = "manual_projected_gripper_contact"


def _metadata(root: Path) -> list[tuple[dict[str, Any], Path]]:
    records: list[tuple[dict[str, Any], Path]] = []
    for path in root.glob("*/metadata.json"):
        records.append((json.loads(path.read_text()), path.parent))
    return records


def _latest_appearance(object_dir: Path, *, manual: bool) -> tuple[dict[str, Any], Path]:
    candidates = [
        (metadata, run_dir)
        for metadata, run_dir in _metadata(object_dir / "roughness")
        if (metadata.get("scoring", {}).get("strategy") == MANUAL_STRATEGY) == manual
    ]
    if not candidates:
        raise RuntimeError(
            f"Missing {'manual' if manual else 'original'} run for {object_dir.name}"
        )
    return max(candidates, key=lambda item: str(item[0].get("created_at", "")))


def _matching_topography(
    object_dir: Path,
    appearance_run_id: str,
) -> tuple[dict[str, Any], Path]:
    candidates = [
        (metadata, run_dir)
        for metadata, run_dir in _metadata(object_dir / "topography")
        if metadata.get("source", {}).get("appearance_run_id") == appearance_run_id
    ]
    if not candidates:
        raise RuntimeError(f"Missing matching topography for {appearance_run_id}")
    return max(candidates, key=lambda item: str(item[0].get("created_at", "")))


def _warning_text(metadata: dict[str, Any]) -> str:
    return ";".join(str(value) for value in metadata.get("quality", {}).get("warnings", []))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Matforcedata")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    dataset_root = args.root / "data" / args.dataset
    objects_root = dataset_root / "objects"
    rows: list[dict[str, Any]] = []
    for object_dir in sorted(path for path in objects_root.iterdir() if path.is_dir()):
        manual, manual_dir = _latest_appearance(object_dir, manual=True)
        original, original_dir = _latest_appearance(object_dir, manual=False)
        manual_topo, manual_topo_dir = _matching_topography(object_dir, manual["run_id"])
        original_topo, _ = _matching_topography(object_dir, original["run_id"])
        spec = json.loads((object_dir / "manual_grasp" / "spec.json").read_text())
        manual_angle = manual_topo["topographic_roughness"]["angle_degrees"]
        original_angle = original_topo["topographic_roughness"]["angle_degrees"]
        uncertainty = manual_topo.get("normal_uncertainty")
        row = {
            "object_id": object_dir.name,
            "grasp_axis": spec["grasp_axis"],
            "contact_pixels": manual["scoring"]["scoring_pixels"],
            "appearance_mean": manual["roughness"]["mean"],
            "appearance_median": manual["roughness"]["median"],
            "appearance_std": manual["roughness"]["std"],
            "original_appearance_mean": original["roughness"]["mean"],
            "appearance_mean_change": (manual["roughness"]["mean"] - original["roughness"]["mean"]),
            "topographic_score_0_1": manual_topo["topographic_roughness"]["score_0_1"],
            "p75_angle_deg": manual_angle["p75"],
            "p90_angle_deg": manual_angle["p90"],
            "p95_angle_deg": manual_angle["p95"],
            "original_topographic_score_0_1": original_topo["topographic_roughness"]["score_0_1"],
            "topographic_score_change": (
                manual_topo["topographic_roughness"]["score_0_1"]
                - original_topo["topographic_roughness"]["score_0_1"]
            ),
            "original_p75_angle_deg": original_angle["p75"],
            "normal_uncertainty_mean": None if uncertainty is None else uncertainty["mean"],
            "appearance_warnings": _warning_text(manual),
            "topography_warnings": _warning_text(manual_topo),
            "mask_rationale": spec["rationale"],
            "appearance_run_dir": str(manual_dir),
            "topography_run_dir": str(manual_topo_dir),
            "original_appearance_run_dir": str(original_dir),
        }
        rows.append(row)

    json_path = dataset_root / "manual_grasp_results.json"
    csv_path = dataset_root / "manual_grasp_results.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"objects": len(rows), "json": str(json_path), "csv": str(csv_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
