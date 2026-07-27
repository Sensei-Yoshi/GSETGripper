"""Create reviewed, object-specific projected gripper-contact masks.

The source images are overhead views, while the real parallel fingers touch opposing
side surfaces.  These masks therefore mark the visible, slightly inset projections of
the two most plausible contact patches.  End caps, openings, necks, cores, and labels
that would not normally be selected as contact surfaces are excluded where possible.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw

ShapeKind = Literal["ellipse", "polygon", "rounded_rectangle"]


@dataclass(frozen=True)
class Shape:
    kind: ShapeKind
    points: tuple[tuple[float, float], ...]
    radius: float = 0.0


@dataclass(frozen=True)
class GraspMaskSpec:
    grasp_axis: str
    rationale: str
    shapes: tuple[Shape, ...]
    clip_to_automatic_foreground: bool = True


def _box(x0: float, y0: float, x1: float, y1: float, *, radius: float = 0.0) -> Shape:
    return Shape("rounded_rectangle", ((x0, y0), (x1, y1)), radius)


def _ellipse(x0: float, y0: float, x1: float, y1: float) -> Shape:
    return Shape("ellipse", ((x0, y0), (x1, y1)))


def _polygon(*points: tuple[float, float]) -> Shape:
    return Shape("polygon", points)


# Coordinates are normalized fractions of the saved Marigold processing crop.  Every
# mask contains the two opposing projected patches, not the volume between the fingers.
SPECS: dict[str, GraspMaskSpec] = {
    "airpods": GraspMaskSpec(
        "left-right",
        "Opposing inset patches on the uniform plastic case; hinge and outer silhouette excluded.",
        (_ellipse(0.22, 0.32, 0.38, 0.69), _ellipse(0.64, 0.32, 0.80, 0.69)),
    ),
    "beaker": GraspMaskSpec(
        "top-bottom",
        "Opposing patches on the cylindrical wall; open rim and irregular handle/base region excluded.",
        (_box(0.28, 0.25, 0.57, 0.37, radius=0.025), _box(0.28, 0.58, 0.57, 0.70, radius=0.025)),
        False,
    ),
    "camera_cardboard_box": GraspMaskSpec(
        "top-bottom",
        "Opposing brown-cardboard edge regions; the printed product label is excluded.",
        (
            _polygon((0.29, 0.20), (0.80, 0.27), (0.79, 0.31), (0.30, 0.28)),
            _polygon((0.29, 0.69), (0.77, 0.70), (0.75, 0.76), (0.29, 0.73)),
        ),
    ),
    "cardboard_bottle_box": GraspMaskSpec(
        "top-bottom",
        "Long opposing cardboard-wall patches away from both end faces.",
        (_box(0.25, 0.28, 0.79, 0.39, radius=0.018), _box(0.25, 0.59, 0.79, 0.70, radius=0.018)),
    ),
    "curved_water_bottle": GraspMaskSpec(
        "top-bottom",
        "Opposing patches on the broad blue body; metal cap and tapered neck excluded.",
        (_box(0.45, 0.25, 0.76, 0.38, radius=0.03), _box(0.45, 0.58, 0.76, 0.71, radius=0.03)),
    ),
    "deodorant": GraspMaskSpec(
        "top-bottom",
        "Opposing patches on the main container body, avoiding the rounded cap and far end.",
        (_box(0.49, 0.25, 0.78, 0.38, radius=0.025), _box(0.49, 0.61, 0.78, 0.74, radius=0.025)),
    ),
    "failed_3d_print": GraspMaskSpec(
        "top-bottom",
        "Opposing patches on the thick rear body where the tapered print can be held stably.",
        (
            _polygon((0.48, 0.25), (0.79, 0.24), (0.76, 0.36), (0.48, 0.38)),
            _polygon((0.48, 0.49), (0.76, 0.45), (0.73, 0.59), (0.48, 0.62)),
        ),
    ),
    "foam_roller": GraspMaskSpec(
        "top-bottom",
        "Opposing foam-barrel patches; axial hole and rounded end regions excluded.",
        (_box(0.42, 0.20, 0.80, 0.34, radius=0.025), _box(0.42, 0.60, 0.80, 0.74, radius=0.025)),
    ),
    "headphones_case": GraspMaskSpec(
        "left-right",
        "Opposing inset fabric-shell patches; zipper seam and silhouette excluded.",
        (_ellipse(0.24, 0.32, 0.40, 0.70), _ellipse(0.67, 0.32, 0.83, 0.70)),
    ),
    "lechee": GraspMaskSpec(
        "left-right",
        "Opposing inset fruit-skin patches that retain the raised cells while avoiding silhouette artifacts.",
        (_ellipse(0.23, 0.34, 0.42, 0.67), _ellipse(0.58, 0.34, 0.77, 0.67)),
    ),
    "orange": GraspMaskSpec(
        "left-right",
        "Opposing inset peel patches representative of the two finger contacts.",
        (_ellipse(0.23, 0.34, 0.42, 0.67), _ellipse(0.58, 0.34, 0.77, 0.67)),
    ),
    "plastic_cup": GraspMaskSpec(
        "top-bottom",
        "Opposing patches on the tapered cup wall; open rim and narrow base excluded.",
        (_box(0.45, 0.31, 0.72, 0.43, radius=0.025), _box(0.45, 0.57, 0.72, 0.69, radius=0.025)),
    ),
    "plastic_water_bottle": GraspMaskSpec(
        "top-bottom",
        "Opposing patches on the central bottle wall; cap, neck, and bottom excluded.",
        (_box(0.43, 0.24, 0.72, 0.37, radius=0.025), _box(0.43, 0.60, 0.72, 0.73, radius=0.025)),
    ),
    "red_bull": GraspMaskSpec(
        "top-bottom",
        "Opposing patches on the straight cylindrical sidewall; pull-tab top and base excluded.",
        (_box(0.39, 0.22, 0.71, 0.35, radius=0.02), _box(0.39, 0.61, 0.71, 0.74, radius=0.02)),
    ),
    "small_3d_print": GraspMaskSpec(
        "top-bottom",
        "Opposing patches on the ribbed outer body; central hole and end silhouette excluded.",
        (_box(0.46, 0.20, 0.80, 0.34, radius=0.018), _box(0.46, 0.64, 0.80, 0.78, radius=0.018)),
    ),
    "soda_can": GraspMaskSpec(
        "top-bottom",
        "Opposing patches on the straight can sidewall; visible can top and bottom shoulder excluded.",
        (_box(0.40, 0.22, 0.71, 0.35, radius=0.02), _box(0.40, 0.61, 0.71, 0.74, radius=0.02)),
    ),
    "toilet_paper": GraspMaskSpec(
        "top-bottom",
        "Opposing patches on the compressible outer paper roll; core and end face excluded.",
        (_box(0.42, 0.20, 0.75, 0.34, radius=0.025), _box(0.42, 0.63, 0.75, 0.77, radius=0.025)),
    ),
}


def _latest_run(object_dir: Path) -> Path:
    candidates: list[tuple[str, Path]] = []
    for metadata_path in (object_dir / "roughness").glob("*/metadata.json"):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("scoring", {}).get("strategy") == "manual_projected_gripper_contact":
            continue
        candidates.append((str(metadata.get("created_at", "")), metadata_path.parent))
    if not candidates:
        raise FileNotFoundError(f"No original appearance run for {object_dir.name}")
    return max(candidates)[1]


def _scaled_points(shape: Shape, size: tuple[int, int]) -> list[tuple[int, int]]:
    width, height = size
    return [(round(x * (width - 1)), round(y * (height - 1))) for x, y in shape.points]


def render_mask(spec: GraspMaskSpec, size: tuple[int, int]) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for shape in spec.shapes:
        points = _scaled_points(shape, size)
        if shape.kind == "ellipse":
            draw.ellipse((*points[0], *points[1]), fill=255)
        elif shape.kind == "rounded_rectangle":
            radius = round(shape.radius * min(size))
            draw.rounded_rectangle((*points[0], *points[1]), radius=radius, fill=255)
        else:
            draw.polygon(points, fill=255)
    return mask


def _overlay(crop: Image.Image, mask: Image.Image) -> Image.Image:
    base = crop.convert("RGB").resize(mask.size, Image.Resampling.LANCZOS)
    rgb = np.asarray(base, dtype=np.float32)
    selected = np.asarray(mask, dtype=np.uint8) >= 128
    tint = np.array([25.0, 230.0, 70.0], dtype=np.float32)
    rgb[selected] = rgb[selected] * 0.42 + tint * 0.58
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Matforcedata")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    objects_root = args.root / "data" / args.dataset / "objects"
    actual = {path.name for path in objects_root.iterdir() if path.is_dir()}
    missing = actual.difference(SPECS)
    unknown = set(SPECS).difference(actual)
    if missing or unknown:
        raise SystemExit(
            f"Mask coverage mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )

    report: list[dict[str, object]] = []
    for object_id in sorted(SPECS):
        object_dir = objects_root / object_id
        run_dir = _latest_run(object_dir)
        metadata = json.loads((run_dir / "metadata.json").read_text())
        size = tuple(int(value) for value in metadata["model"]["processed_size"])
        spec = SPECS[object_id]
        mask = render_mask(spec, size)
        if spec.clip_to_automatic_foreground:
            foreground_name = metadata["artifacts"]["inference_mask"]
            with Image.open(run_dir / foreground_name) as foreground_source:
                foreground = foreground_source.convert("L").resize(
                    size,
                    Image.Resampling.NEAREST,
                )
            mask = Image.fromarray(
                np.minimum(
                    np.asarray(mask, dtype=np.uint8),
                    np.asarray(foreground, dtype=np.uint8),
                ),
                mode="L",
            )
        pixels = int((np.asarray(mask) >= 128).sum())
        if pixels < 256:
            raise RuntimeError(f"Manual mask for {object_id} is too small: {pixels} pixels")
        output_dir = object_dir / "manual_grasp"
        output_dir.mkdir(parents=True, exist_ok=True)
        mask.save(output_dir / "scoring_mask.png")
        with Image.open(run_dir / metadata["artifacts"]["inference_crop"]) as crop:
            _overlay(crop, mask).save(output_dir / "scoring_overlay.png")
        record = {
            "object_id": object_id,
            "source_run_id": metadata["run_id"],
            "processed_size": list(size),
            "scoring_pixels": pixels,
            "scoring_fraction_of_frame": pixels / (size[0] * size[1]),
            "grasp_axis": spec.grasp_axis,
            "rationale": spec.rationale,
            "coordinate_system": "normalized Marigold processing crop",
            "clipped_to_automatic_foreground": spec.clip_to_automatic_foreground,
            "mask": "scoring_mask.png",
            "overlay": "scoring_overlay.png",
        }
        (output_dir / "spec.json").write_text(json.dumps(record, indent=2) + "\n")
        report.append(record)
        print(json.dumps(record), flush=True)
    (objects_root.parent / "manual_grasp_masks.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
