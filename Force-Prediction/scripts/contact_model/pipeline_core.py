"""Shared contact-analysis core used by both the CLI (capture_and_analyze.py)
and the Streamlit test page. One code path so the two cannot drift.

``analyze_image`` takes a saved image file plus a ``ContactParams`` bundle,
runs outline extraction + the contact model, writes every artifact into
``run_dir`` (spline overlay, CSV, SVG, cutout, mask, contact figure,
summary.json), optionally appends a master index.csv, and returns the
estimate and the summary dict.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import extract_object_outline as outline
import numpy as np
from contact_area import ContactEstimate, FingerGeometry, estimate_contact
from viz import plot_estimate


@dataclass
class ContactParams:
    """Every knob the contact model needs, with paper-default values."""

    px_per_mm: float
    object_type: str = "prismatic"          # or "axisymmetric"
    closing_axis: str = "x"                  # jaw closing axis in the image
    k_max: float = 2.0                       # 1/mm
    delta: float = 0.3                       # mm
    L: float = 4.0                           # pad length, mm
    w_pad: float = 12.0                      # pad width, mm
    ds: float = 0.25                         # resample step, mm
    smoothing: float = 0.2                   # pre-differentiation smoothing, mm
    sweep_k: tuple[float, ...] = (1.0, 2.0, 4.0)
    finger: FingerGeometry | None = None     # set -> drop-depth model
    # Pad hangs from the object top: contact band = top L, below disregarded.
    # Ignored when ``finger`` is set (drop-depth wins).
    pad_top_anchored: bool = True


def outline_csv_to_mm(
    csv_path: Path, px_per_mm: float, closing_axis: str
) -> np.ndarray:
    """Load a spline_points.csv (px, image y-down) as mm points, y-up."""
    with csv_path.open() as f:
        reader = csv.reader(f)
        next(reader)
        pts_px = np.array([[float(x), float(y)] for x, y in reader])
    pts_mm = pts_px / px_per_mm
    pts_mm[:, 1] = pts_mm[:, 1].max() - pts_mm[:, 1]  # image y-down -> y-up
    if closing_axis == "y":
        pts_mm = pts_mm[:, ::-1].copy()
    return pts_mm


def _finger_summary(f) -> dict:
    return {
        "contact_mm": round(f.contact_length, 3),
        "window_mm": round(f.window_length, 3),
        "area_mm2": round(f.area, 3),
        "fraction": round(f.fraction, 4),
    }


def build_summary(
    name: str, image_name: str, est: ContactEstimate, p: ContactParams,
    sweep: dict[str, float],
) -> dict:
    finger = p.finger
    pts = est.boundary.pts
    object_height_mm = float(pts[:, 1].max() - pts[:, 1].min())
    object_width_mm = float(pts[:, 0].max() - pts[:, 0].min())
    return {
        "name": name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "image": image_name,
        "px_per_mm": p.px_per_mm,
        "params": {
            "k_max_per_mm": p.k_max, "delta_mm": p.delta, "L_mm": p.L,
            "w_pad_mm": p.w_pad, "ds_mm": p.ds, "smoothing_mm": p.smoothing,
            "object_type": p.object_type, "closing_axis": p.closing_axis,
        },
        "finger": None if finger is None else {
            "finger_length_mm": finger.finger_length,
            "pad_length_mm": finger.pad_length,
            "pad_start_mm": finger.pad_start,
            "tip_clearance_mm": finger.tip_clearance,
            "palm_standoff_mm": finger.palm_standoff,
            "tip_height_mm": round(est.tip_height, 2),
            "pad_band_mm": [round(v, 2) for v in est.pad_band],
            "grasp_feasible": bool(est.feasible),
        },
        "results": {
            "object_height_mm": round(object_height_mm, 2),
            "object_width_mm": round(object_width_mm, 2),
            "perimeter_mm": round(est.boundary.length, 2),
            "antipodal_grasp": bool(est.pair.antipodal),
            "left": _finger_summary(est.left),
            "right": _finger_summary(est.right),
            "total_area_mm2": round(est.total_area, 3),
            "mean_fraction": round(est.mean_fraction, 4),
        },
        "k_max_sweep_mean_fraction": sweep,
    }


def append_index(index_csv: Path, name: str, est: ContactEstimate,
                 p: ContactParams, summary: dict, folder: str) -> None:
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not index_csv.exists()
    with index_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow([
                "timestamp", "name", "px_per_mm", "k_max", "delta", "L",
                "w_pad", "object_type", "object_height_mm", "object_width_mm",
                "feasible", "contact_L_mm", "contact_R_mm", "total_area_mm2",
                "mean_fraction", "antipodal", "folder",
            ])
        w.writerow([
            summary["timestamp"], name, p.px_per_mm, p.k_max, p.delta, p.L,
            p.w_pad, p.object_type,
            summary["results"]["object_height_mm"],
            summary["results"]["object_width_mm"], bool(est.feasible),
            summary["results"]["left"]["contact_mm"],
            summary["results"]["right"]["contact_mm"],
            summary["results"]["total_area_mm2"],
            summary["results"]["mean_fraction"],
            bool(est.pair.antipodal), folder,
        ])


def analyze_image(
    image_path: Path,
    run_dir: Path,
    name: str,
    params: ContactParams,
    session=None,
    index_csv: Path | None = None,
) -> tuple[ContactEstimate, dict, dict[str, Path]]:
    """Full pipeline for one saved image. Returns (estimate, summary, paths).

    ``paths`` maps logical artifact names to files for easy display.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    outputs = outline.process_image(image_path, run_dir, session=session)
    csv_path = run_dir / f"{stem}_spline_points.csv"
    if csv_path not in outputs:
        raise RuntimeError(f"extractor did not produce {csv_path}")

    pts_mm = outline_csv_to_mm(csv_path, params.px_per_mm, params.closing_axis)

    est = estimate_contact(
        pts_mm, k_max=params.k_max, delta=params.delta, L=params.L,
        w_pad=params.w_pad, ds=params.ds, smoothing_mm=params.smoothing,
        object_type=params.object_type, finger=params.finger,
        pad_top_anchored=params.pad_top_anchored,
    )

    fig_path = run_dir / f"{stem}_contact.png"
    plot_estimate(
        est, fig_path,
        f"{name}  (k_max={params.k_max}/mm, delta={params.delta} mm, "
        f"L={params.L} mm, {params.object_type})",
    )

    sweep = {}
    for k in params.sweep_k:
        e = est if abs(k - params.k_max) < 1e-12 else estimate_contact(
            pts_mm, k_max=k, delta=params.delta, L=params.L,
            w_pad=params.w_pad, ds=params.ds, smoothing_mm=params.smoothing,
            object_type=params.object_type, finger=params.finger,
            pad_top_anchored=params.pad_top_anchored,
        )
        sweep[f"{k:g}"] = round(e.mean_fraction, 4)

    summary = build_summary(name, image_path.name, est, params, sweep)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if index_csv is not None:
        append_index(index_csv, name, est, params, summary, run_dir.name)

    paths = {
        "raw": image_path,
        "cutout": run_dir / f"{stem}_cutout.png",
        "mask": run_dir / f"{stem}_mask.png",
        "spline_overlay": run_dir / f"{stem}_spline_overlay.png",
        "spline_csv": csv_path,
        "spline_svg": run_dir / f"{stem}_spline.svg",
        "contact_fig": fig_path,
        "summary": run_dir / "summary.json",
    }
    return est, summary, paths
