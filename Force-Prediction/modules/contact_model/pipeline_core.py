"""Shared image-to-contact-fraction pipeline for CLI and Streamlit."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import extract_object_outline as outline
from .contact_area import ContactEstimate, estimate_contact
from .viz import plot_estimate

SUMMARY_SCHEMA_VERSION = 2


@dataclass
class ContactParams:
    """Calibrated knobs for projected two-pad contact estimation."""

    px_per_mm: float
    closing_axis: str = "x"
    mode: str = "compliant"
    pad_length_mm: float = 106.68
    minimum_bend_radius_mm: float = 20.0
    side_angle_deg: float = 30.0
    minimum_contact_fraction: float = 0.05
    rigid_contact_tolerance_mm: float = 0.5
    finger_extension_mm: float = 63.5
    planar_threshold: float = 0.15
    ds: float = 0.25
    smoothing: float = 0.2
    sweep_radii_mm: tuple[float, ...] = (10.0, 20.0, 30.0)


def outline_csv_to_mm(
    csv_path: Path, px_per_mm: float, closing_axis: str
) -> np.ndarray:
    """Load a spline CSV in image pixels as millimetres with y pointing up."""
    with csv_path.open() as f:
        reader = csv.reader(f)
        next(reader)
        pts_px = np.array([[float(x), float(y)] for x, y in reader])
    pts_mm = pts_px / px_per_mm
    pts_mm[:, 1] = pts_mm[:, 1].max() - pts_mm[:, 1]
    if closing_axis == "y":
        pts_mm = pts_mm[:, ::-1].copy()
    return pts_mm


def _finger_summary(f) -> dict:
    return {
        "contact_length_mm": round(f.contact_length, 3),
        "pad_length_mm": round(f.pad_length, 3),
        "contact_fraction": round(f.fraction, 4),
    }


def build_summary(
    name: str,
    image_name: str,
    est: ContactEstimate,
    params: ContactParams,
    sweep: dict[str, float],
) -> dict:
    pts = est.boundary.pts
    params_out: dict = {
        "mode": params.mode,
        "pad_length_mm": params.pad_length_mm,
        "side_angle_deg": params.side_angle_deg,
        "minimum_contact_fraction": params.minimum_contact_fraction,
        "ds_mm": params.ds,
        "smoothing_mm": params.smoothing,
        "closing_axis": params.closing_axis,
    }
    if params.mode == "rigid":
        params_out["rigid_contact_tolerance_mm"] = params.rigid_contact_tolerance_mm
        params_out["finger_extension_mm"] = params.finger_extension_mm
        params_out["planar_threshold"] = params.planar_threshold
        params_out["placement"] = "maximized_contact_area"
    else:
        params_out["minimum_bend_radius_mm"] = params.minimum_bend_radius_mm
        params_out["k_max_per_mm"] = round(1.0 / params.minimum_bend_radius_mm, 8)
        params_out["placement"] = "pad_top_aligned_to_object_top"

    results = {
        "object_height_mm": round(float(np.ptp(pts[:, 1])), 2),
        "object_width_mm": round(float(np.ptp(pts[:, 0])), 2),
        "perimeter_mm": round(est.boundary.length, 2),
        "grasp_feasible": bool(est.feasible),
        "antipodal_grasp": bool(est.pair.antipodal),
        "chosen_pad_top_mm": round(float(est.pad_band[1]), 3),
        "left": _finger_summary(est.left),
        "right": _finger_summary(est.right),
        "combined_contact_length_mm": round(est.combined_contact_length, 3),
        "geometric_contact_fraction": round(est.geometric_contact_fraction, 4),
        "contact_floor_applied": bool(est.contact_floor_applied),
        "combined_contact_fraction": round(est.combined_contact_fraction, 4),
    }
    if params.mode == "rigid":
        results["is_planar"] = bool(est.is_planar)

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "metric": "projected_two_pad_contact_fraction",
        "metric_definition": "max(minimum_contact_fraction, geometric_contact_fraction) for an antipodal grasp; otherwise 0",
        "name": name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "image": image_name,
        "px_per_mm": params.px_per_mm,
        "params": params_out,
        "results": results,
        "bend_radius_sweep_combined_fraction": sweep,
    }


def analyze_image(
    image_path: Path,
    run_dir: Path,
    name: str,
    params: ContactParams,
    session=None,
) -> tuple[ContactEstimate, dict, dict[str, Path]]:
    """Extract an outline, estimate contact fraction, and save v2 artifacts."""
    run_dir.mkdir(parents=True, exist_ok=True)
    outputs = outline.process_image(image_path, run_dir, session=session)
    csv_path = run_dir / "spline_points.csv"
    if csv_path not in outputs:
        raise RuntimeError(f"extractor did not produce {csv_path}")

    pts_mm = outline_csv_to_mm(
        csv_path, params.px_per_mm, params.closing_axis
    )
    estimate_kwargs: dict[str, Any] = {
        "mode": params.mode,
        "pad_length_mm": params.pad_length_mm,
        "side_angle_deg": params.side_angle_deg,
        "minimum_contact_fraction": params.minimum_contact_fraction,
        "rigid_contact_tolerance_mm": params.rigid_contact_tolerance_mm,
        "finger_extension_mm": params.finger_extension_mm,
        "planar_threshold": params.planar_threshold,
        "ds": params.ds,
        "smoothing_mm": params.smoothing,
    }
    est = estimate_contact(
        pts_mm,
        minimum_bend_radius_mm=params.minimum_bend_radius_mm,
        **estimate_kwargs,
    )

    if params.mode == "rigid":
        title = (
            f"{name}  (rigid, tol={params.rigid_contact_tolerance_mm:g} mm, "
            f"side tolerance={params.side_angle_deg:g} deg, "
            f"L={params.pad_length_mm:g} mm)"
        )
    else:
        title = (
            f"{name}  (R_min={params.minimum_bend_radius_mm:g} mm, "
            f"side tolerance={params.side_angle_deg:g} deg, "
            f"L={params.pad_length_mm:g} mm)"
        )
    fig_path = run_dir / "contact.png"
    plot_estimate(est, fig_path, title)

    # The bend-radius sweep is a compliant-only sensitivity study; a rigid
    # finger has no bend radius, so leave it empty.
    sweep: dict[str, float] = {}
    if params.mode != "rigid":
        for radius in params.sweep_radii_mm:
            candidate = est if abs(radius - params.minimum_bend_radius_mm) < 1e-12 else estimate_contact(
                pts_mm,
                minimum_bend_radius_mm=radius,
                **estimate_kwargs,
            )
            sweep[f"{radius:g}"] = round(candidate.combined_contact_fraction, 4)

    summary = build_summary(name, image_path.name, est, params, sweep)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    paths = {
        "raw": image_path,
        "cutout": run_dir / "cutout.png",
        "mask": run_dir / "mask.png",
        "spline_overlay": run_dir / "spline_overlay.png",
        "spline_csv": csv_path,
        "spline_svg": run_dir / "spline.svg",
        "contact_fig": fig_path,
        "summary": summary_path,
    }
    return est, summary, paths
