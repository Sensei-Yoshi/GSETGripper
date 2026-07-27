"""Re-run Marigold appearance and normal analysis with manual grasp masks."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.config import load_config  # noqa: E402
from modules.models.background_remover import BackgroundRemover  # noqa: E402
from modules.models.marigold import (  # noqa: E402
    MarigoldAnalyzer,
    available_device,
    list_saved_runs,
    run_marigold,
)
from modules.models.topographic_roughness import (  # noqa: E402
    MarigoldNormalsAnalyzer,
    run_topographic_roughness,
)

MANUAL_STRATEGY = "manual_projected_gripper_contact"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Matforcedata")
    parser.add_argument("--objects", nargs="*")
    parser.add_argument(
        "--stage",
        choices=("appearance", "topography", "all"),
        default="all",
    )
    return parser


def _source_appearance_run(object_dir: Path, *, require_manual: bool) -> dict[str, object]:
    runs = list_saved_runs(object_dir / "roughness")
    for run in runs:
        strategy = run.get("scoring", {}).get("strategy")
        if (strategy == MANUAL_STRATEGY) == require_manual:
            return run
    kind = "manual" if require_manual else "original"
    raise RuntimeError(f"No {kind} appearance run found for {object_dir.name}")


def _clear_accelerator() -> None:
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main() -> int:
    args = _parser().parse_args()
    cfg = load_config()
    objects_root = cfg.root / "data" / args.dataset / "objects"
    available = sorted(
        path.name
        for path in objects_root.iterdir()
        if path.is_dir() and (path / "manual_grasp" / "scoring_mask.png").is_file()
    )
    requested = sorted(set(args.objects or available))
    unknown = set(requested).difference(available)
    if unknown:
        raise SystemExit(f"Objects without a manual mask: {', '.join(sorted(unknown))}")

    device = available_device()
    appearance_results: list[dict[str, object]] = []
    if args.stage in {"appearance", "all"}:
        analyzer = MarigoldAnalyzer(device=device, processing_resolution=768)
        remover = BackgroundRemover(model_name="isnet-general-use")
        for index, object_id in enumerate(requested, 1):
            object_dir = objects_root / object_id
            source = _source_appearance_run(object_dir, require_manual=False)
            source_dir = Path(str(source["run_dir"]))
            artifacts = source["artifacts"]
            with Image.open(source_dir / str(artifacts["input"])) as image:
                input_image = image.convert("RGB").copy()
            with Image.open(object_dir / "manual_grasp" / "scoring_mask.png") as image:
                scoring_mask = image.convert("L").copy()
            spec = json.loads((object_dir / "manual_grasp" / "spec.json").read_text())
            result = run_marigold(
                analyzer,
                input_image,
                object_dir / "roughness",
                background_remover=remover,
                source_label=object_id,
                dataset_id=args.dataset,
                object_id=object_id,
                source_path=str(source.get("source", {}).get("path") or ""),
                num_inference_steps=4,
                ensemble_size=3,
                seed=20260721,
                scoring_mask_source=scoring_mask,
                scoring_mask_rationale=str(spec["rationale"]),
            )
            summary = {
                "object_id": object_id,
                "appearance_mean": result["roughness"]["mean"],
                "appearance_median": result["roughness"]["median"],
                "quality": result["quality"]["status"],
                "run_dir": result["run_dir"],
            }
            appearance_results.append(summary)
            print(f"[appearance {index}/{len(requested)}] {json.dumps(summary)}", flush=True)
        del analyzer, remover
        _clear_accelerator()

    topography_results: list[dict[str, object]] = []
    if args.stage in {"topography", "all"}:
        analyzer = MarigoldNormalsAnalyzer(device=device, processing_resolution=768)
        for index, object_id in enumerate(requested, 1):
            object_dir = objects_root / object_id
            appearance = _source_appearance_run(object_dir, require_manual=True)
            result = run_topographic_roughness(
                analyzer,
                Path(str(appearance["run_dir"])),
                object_dir / "topography",
                num_inference_steps=4,
                ensemble_size=3,
                seed=20260721,
                base_surface_sigma_ratio=0.04,
            )
            topo = result["topographic_roughness"]
            summary = {
                "object_id": object_id,
                "topographic_score_0_1": topo["score_0_1"],
                "p75_angle_deg": topo["angle_degrees"]["p75"],
                "quality": result["quality"]["status"],
                "run_dir": result["run_dir"],
            }
            topography_results.append(summary)
            print(f"[topography {index}/{len(requested)}] {json.dumps(summary)}", flush=True)

    output = {
        "dataset_id": args.dataset,
        "mask_strategy": MANUAL_STRATEGY,
        "appearance": appearance_results,
        "topography": topography_results,
    }
    (objects_root.parent / f"manual_grasp_{args.stage}_run.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
