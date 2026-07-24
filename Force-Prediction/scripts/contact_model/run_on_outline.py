"""Run the contact model on a real extracted outline.

Consumes the ``*_spline_points.csv`` written by extract_object_outline.py
(pixel coordinates, image y-down) and produces the contact estimate plus an
overlay figure.

    /Users/premshah/Desktop/Robotics/GSET/env/bin/python \
        scripts/contact_model/run_on_outline.py \
        path/to/object_spline_points.csv --px-per-mm 8.4 \
        --object-type axisymmetric --k-max 2.0 --w-pad 12

--px-per-mm is REQUIRED and must come from a fiducial in the scene (object
of known width, or the jaw opening at capture). Without it every mm-valued
parameter (k_max, delta, L) is meaningless.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contact_area import estimate_contact
from viz import plot_estimate


def load_csv(path: Path) -> np.ndarray:
    with path.open() as f:
        reader = csv.reader(f)
        next(reader)  # header
        return np.array([[float(x), float(y)] for x, y in reader])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--px-per-mm", type=float, required=True)
    ap.add_argument("--object-type", choices=["prismatic", "axisymmetric"],
                    default="prismatic")
    ap.add_argument("--closing-axis", choices=["x", "y"], default="x",
                    help="jaw closing axis in the image")
    ap.add_argument("--k-max", type=float, default=2.0, help="1/mm")
    ap.add_argument("--delta", type=float, default=0.3, help="mm")
    ap.add_argument("--L", type=float, default=4.0, help="pad length, mm")
    ap.add_argument("--w-pad", type=float, default=12.0, help="pad width, mm")
    ap.add_argument("--ds", type=float, default=0.25, help="mm")
    ap.add_argument("--smoothing", type=float, default=0.2,
                    help="mm smoothing before differentiating")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    pts_px = load_csv(args.csv)
    pts_mm = pts_px / args.px_per_mm
    pts_mm[:, 1] = pts_mm[:, 1].max() - pts_mm[:, 1]  # image y-down -> y-up
    if args.closing_axis == "y":
        pts_mm = pts_mm[:, ::-1].copy()  # mirror; orientation re-fixed later

    est = estimate_contact(
        pts_mm, k_max=args.k_max, delta=args.delta, L=args.L,
        w_pad=args.w_pad, ds=args.ds, smoothing_mm=args.smoothing,
        object_type=args.object_type,
    )

    out = args.out or args.csv.parent
    fig_path = out / f"{args.csv.stem}_contact.png"
    plot_estimate(est, fig_path, args.csv.stem)

    print(f"object            : {args.csv.stem}")
    print(f"perimeter         : {est.boundary.length:.1f} mm")
    print(f"antipodal grasp   : {est.pair.antipodal}")
    for f in (est.left, est.right):
        print(f"{f.side:5s} finger      : contact {f.contact_length:.2f} mm "
              f"of {f.window_length:.2f} mm window, area {f.area:.2f} mm^2, "
              f"fraction {f.fraction:.3f}")
    print(f"total area        : {est.total_area:.2f} mm^2")
    print(f"mean fraction     : {est.mean_fraction:.3f}")
    print(f"figure            : {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
