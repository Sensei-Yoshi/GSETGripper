"""Run the projected contact-fraction model on an extracted outline CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .contact_area import estimate_contact
from .viz import plot_estimate


def load_csv(path: Path) -> np.ndarray:
    with path.open() as file:
        reader = csv.reader(file)
        next(reader)
        return np.array([[float(x), float(y)] for x, y in reader])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--px-per-mm", type=float, required=True)
    parser.add_argument("--closing-axis", choices=["x", "y"], default="x")
    parser.add_argument("--mode", choices=["compliant", "rigid"], default="compliant")
    parser.add_argument("--pad-length-mm", type=float, default=106.68)
    parser.add_argument("--minimum-bend-radius-mm", type=float, default=20.0)
    parser.add_argument("--side-angle-deg", type=float, default=30.0)
    parser.add_argument("--minimum-contact-fraction", type=float, default=0.05)
    parser.add_argument("--rigid-contact-tolerance-mm", type=float, default=0.5)
    parser.add_argument("--finger-extension-mm", type=float, default=63.5)
    parser.add_argument("--planar-threshold", type=float, default=0.15)
    parser.add_argument("--ds", type=float, default=0.25)
    parser.add_argument("--smoothing", type=float, default=0.2)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    points = load_csv(args.csv) / args.px_per_mm
    points[:, 1] = points[:, 1].max() - points[:, 1]
    if args.closing_axis == "y":
        points = points[:, ::-1].copy()

    estimate = estimate_contact(
        points,
        mode=args.mode,
        pad_length_mm=args.pad_length_mm,
        minimum_bend_radius_mm=args.minimum_bend_radius_mm,
        side_angle_deg=args.side_angle_deg,
        minimum_contact_fraction=args.minimum_contact_fraction,
        rigid_contact_tolerance_mm=args.rigid_contact_tolerance_mm,
        finger_extension_mm=args.finger_extension_mm,
        planar_threshold=args.planar_threshold,
        ds=args.ds,
        smoothing_mm=args.smoothing,
    )
    output = args.out or args.csv.parent
    object_name = args.csv.stem.removesuffix("_spline_points")
    figure = output / f"{object_name}_contact_v2.png"
    plot_estimate(estimate, figure, object_name)

    print(f"object          : {args.csv.stem}")
    print(f"mode            : {estimate.mode}")
    if estimate.mode == "rigid":
        print(f"planar          : {estimate.is_planar}")
        print(f"chosen pad top  : {estimate.pad_band[1]:.2f} mm")
    print(f"perimeter       : {estimate.boundary.length:.1f} mm")
    print(f"antipodal       : {estimate.pair.antipodal}")
    print(f"feasible        : {estimate.feasible}")
    for finger in (estimate.left, estimate.right):
        print(
            f"{finger.side:5s} contact    : {finger.contact_length:.2f} mm, "
            f"fraction {finger.fraction:.4f}"
        )
    print(f"geometric length: {estimate.combined_contact_length:.2f} mm")
    print(f"geometric fraction: {estimate.geometric_contact_fraction:.4f}")
    print(f"floor applied   : {estimate.contact_floor_applied}")
    print(f"combined fraction: {estimate.combined_contact_fraction:.4f}")
    print(f"figure          : {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
