"""Capture or load an image and estimate projected two-pad contact fraction.

The primary result is dimensionless:

    (left contact length + right contact length) / (2 * pad length)

Pad width is constant and cancels.  Results use schema v2 and are appended to
``data/real_contact_area/index_v2.csv``; legacy artifacts remain untouched.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contact_area import ContactEstimate
from pipeline_core import ContactParams, analyze_image

DEFAULT_OUT_ROOT = Path(__file__).resolve().parents[2] / "data" / "real_contact_area"
SWEEP_RADII_DEFAULT = "10,20,30"


def preview_and_capture(camera: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {camera}")
    window = "capture  [SPACE = shoot, Q/ESC = quit]"
    frame = None
    try:
        while True:
            ok, live = cap.read()
            if not ok:
                raise SystemExit("camera stopped delivering frames")
            cv2.imshow(window, live)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                frame = live.copy()
                break
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyWindow(window)
        cv2.waitKey(1)
    return frame


def click_scale(frame: np.ndarray, reference_width_mm: float) -> float | None:
    """Click the two reference edges and return pixels per millimetre."""
    window = (
        f"scale: click both edges of the {reference_width_mm:g} mm reference "
        "(ESC = abort)"
    )
    points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):  # noqa: ANN001, ARG001
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            shown = frame.copy()
            for point in points:
                cv2.circle(shown, point, 6, (0, 255, 0), 2)
            if len(points) == 2:
                cv2.line(shown, points[0], points[1], (0, 255, 0), 2)
            cv2.imshow(window, shown)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                return None
            if len(points) == 2:
                cv2.waitKey(400)
                break
    finally:
        cv2.destroyWindow(window)
        cv2.waitKey(1)

    pixels = float(np.hypot(
        points[0][0] - points[1][0], points[0][1] - points[1][1]
    ))
    if pixels < 5:
        print("clicked points nearly coincide; aborting scale", file=sys.stderr)
        return None
    return pixels / reference_width_mm


def show_image(path: Path) -> None:
    image = cv2.imread(str(path))
    if image is None:
        return
    window = f"{path.name}  [any key to continue]"
    cv2.imshow(window, image)
    cv2.waitKey(0)
    cv2.destroyWindow(window)
    cv2.waitKey(1)


def analyze_capture(
    name: str, image_path: Path, run_dir: Path, args: argparse.Namespace
) -> ContactEstimate:
    params = ContactParams(
        px_per_mm=args.px_per_mm,
        closing_axis=args.closing_axis,
        pad_length_mm=args.pad_length_mm,
        minimum_bend_radius_mm=args.minimum_bend_radius_mm,
        side_angle_deg=args.side_angle_deg,
        minimum_contact_fraction=args.minimum_contact_fraction,
        ds=args.ds,
        smoothing=args.smoothing,
        sweep_radii_mm=tuple(
            float(value) for value in args.sweep_radii_mm.split(",")
            if value.strip()
        ),
    )
    estimate, summary, paths = analyze_image(
        image_path,
        run_dir,
        name,
        params,
        index_csv=args.out_root / "index_v2.csv",
    )
    print(
        f"contact L/R : {estimate.left.contact_length:.2f} / "
        f"{estimate.right.contact_length:.2f} mm"
    )
    print(f"combined     : {estimate.combined_contact_fraction:.4f}")
    print(
        "radius sweep: "
        f"{summary['bend_radius_sweep_combined_fraction']}"
    )
    print(f"figure       : {paths['contact_fig']}")
    return estimate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--px-per-mm", type=float, default=None)
    parser.add_argument("--ref-width-mm", type=float, default=None)
    parser.add_argument("--closing-axis", choices=["x", "y"], default="x")
    parser.add_argument("--pad-length-mm", type=float, default=106.68)
    parser.add_argument("--minimum-bend-radius-mm", type=float, default=20.0)
    parser.add_argument("--side-angle-deg", type=float, default=30.0)
    parser.add_argument("--minimum-contact-fraction", type=float, default=0.05)
    parser.add_argument("--ds", type=float, default=0.25)
    parser.add_argument("--smoothing", type=float, default=0.2)
    parser.add_argument("--sweep-radii-mm", default=SWEEP_RADII_DEFAULT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.px_per_mm is None and args.ref_width_mm is None:
        parser.error("scale required: pass --px-per-mm or --ref-width-mm")
    args.out_root.mkdir(parents=True, exist_ok=True)

    def run_one(frame: np.ndarray | None, image_source: Path | None) -> None:
        name = args.name or (image_source.stem if image_source else None)
        if name is None:
            name = input("object name: ").strip() or "object"
        name = name.replace(" ", "_")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = args.out_root / f"{name}_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=False)

        if frame is not None:
            image_path = run_dir / f"{name}.png"
            if not cv2.imwrite(str(image_path), frame):
                raise OSError(f"could not write {image_path}")
        else:
            image = cv2.imread(str(image_source))
            if image is None:
                raise SystemExit(f"could not read {image_source}")
            image_path = run_dir / f"{name}{image_source.suffix}"
            if not cv2.imwrite(str(image_path), image):
                raise OSError(f"could not write {image_path}")

        if args.px_per_mm is None:
            scale_image = frame if frame is not None else cv2.imread(str(image_path))
            args.px_per_mm = click_scale(scale_image, args.ref_width_mm)
            if args.px_per_mm is None:
                print("no scale set; run discarded", file=sys.stderr)
                return

        analyze_capture(name, image_path, run_dir, args)
        if not args.no_show:
            show_image(run_dir / f"{image_path.stem}_contact.png")

    if args.image is not None:
        run_one(None, args.image)
        return 0

    while True:
        frame = preview_and_capture(args.camera)
        if frame is None:
            break
        run_one(frame, None)
        args.name = None
        print("back to preview - SPACE for next object, Q to finish")
        time.sleep(0.3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
