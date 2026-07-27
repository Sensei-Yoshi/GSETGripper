"""Estimate raised bumps and grooves from saved Marigold appearance crops."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.config import load_config  # noqa: E402
from modules.models.marigold import available_device, list_saved_runs  # noqa: E402
from modules.models.topographic_roughness import (  # noqa: E402
    DEFAULT_BASE_SURFACE_SIGMA_RATIO,
    DEFAULT_ENSEMBLE_SIZE,
    DEFAULT_INFERENCE_STEPS,
    DEFAULT_PROCESSING_RESOLUTION,
    MarigoldNormalsAnalyzer,
    run_topographic_roughness,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Matforcedata")
    parser.add_argument(
        "--objects",
        nargs="*",
        help="Object IDs to process; omitted means every object with a roughness run.",
    )
    parser.add_argument("--processing-resolution", type=int, default=DEFAULT_PROCESSING_RESOLUTION)
    parser.add_argument("--inference-steps", type=int, default=DEFAULT_INFERENCE_STEPS)
    parser.add_argument("--ensemble-size", type=int, default=DEFAULT_ENSEMBLE_SIZE)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--base-surface-sigma-ratio",
        type=float,
        default=DEFAULT_BASE_SURFACE_SIGMA_RATIO,
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    cfg = load_config()
    dataset_root = cfg.root / "data" / args.dataset
    objects_root = dataset_root / "objects"
    if not objects_root.is_dir():
        raise SystemExit(f"No prepared object directory exists at {objects_root}")
    available = {
        path.name
        for path in objects_root.iterdir()
        if path.is_dir() and list_saved_runs(path / "roughness")
    }
    requested = set(args.objects or available)
    unknown = requested.difference(available)
    if unknown:
        raise SystemExit(f"Unknown object IDs: {', '.join(sorted(unknown))}")

    jobs = []
    for object_id in sorted(requested):
        object_dir = objects_root / object_id
        runs = list_saved_runs(object_dir / "roughness")
        if runs:
            jobs.append((object_id, Path(runs[0]["run_dir"]), object_dir / "topography"))
    if not jobs:
        raise SystemExit("No saved Marigold roughness runs were found for the requested objects.")

    analyzer = MarigoldNormalsAnalyzer(
        device=available_device(),
        processing_resolution=args.processing_resolution,
    )
    summaries = []
    for index, (object_id, appearance_run, output_root) in enumerate(jobs, 1):
        result = run_topographic_roughness(
            analyzer,
            appearance_run,
            output_root,
            num_inference_steps=args.inference_steps,
            ensemble_size=args.ensemble_size,
            seed=args.seed,
            base_surface_sigma_ratio=args.base_surface_sigma_ratio,
        )
        topography = result["topographic_roughness"]
        summary = {
            "object_id": object_id,
            "score_0_1": topography["score_0_1"],
            "p75_angle_deg": topography["angle_degrees"]["p75"],
            "p90_angle_deg": topography["angle_degrees"]["p90"],
            "quality": result["quality"]["status"],
            "run_dir": result["run_dir"],
        }
        summaries.append(summary)
        print(f"[{index}/{len(jobs)}] {json.dumps(summary)}", flush=True)
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
