"""End-to-end contact-fraction pipeline: SPACE to capture -> analysis folder.

Live camera preview; press SPACE to capture an object, Q/ESC to quit. Each
capture runs background removal + spline extraction + the contact model and
writes one self-contained folder under data/real_contact_area/:

    <name>_<YYYYmmdd-HHMMSS>/
        <name>.png                  raw capture
        <name>_cutout.png           rembg cutout
        <name>_mask.png             foreground mask
        <name>_spline_overlay.png   fitted outline over the photo
        <name>_spline_points.csv    outline points (px)
        <name>_spline.svg           outline vector
        <name>_contact.png          contact visualization + numbers
        summary.json                every parameter and every number
    index.csv                       one row per run, master log

Scale: pass --px-per-mm directly, or pass --ref-width-mm W and click the two
edges of a known-width reference in the frozen capture. Without a real scale
every mm-valued parameter is meaningless.

Examples:
    # live camera, click-to-scale against a 50 mm wide fiducial
    $VENV scripts/contact_model/capture_and_analyze.py --ref-width-mm 50

    # process an existing photo instead of the camera
    $VENV scripts/contact_model/capture_and_analyze.py \
        --image data/MatForce/plastic_cup.png --px-per-mm 8.4 \
        --object-type axisymmetric

Capture tips: object alone against a plain background (rembg keeps the
largest foreground component), jaws closing along the image x axis.
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

from contact_area import ContactEstimate, FingerGeometry
from pipeline_core import ContactParams, analyze_image

DEFAULT_OUT_ROOT = (
    Path(__file__).resolve().parents[2] / "data" / "real_contact_area"
)
SWEEP_K_DEFAULT = "1,2,4"


# ---------------------------------------------------------------------------
# Camera / interaction
# ---------------------------------------------------------------------------


def preview_and_capture(camera: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {camera}")
    win = "capture  [SPACE = shoot, Q/ESC = quit]"
    frame = None
    try:
        while True:
            ok, live = cap.read()
            if not ok:
                raise SystemExit("camera stopped delivering frames")
            cv2.imshow(win, live)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE
                frame = live.copy()
                break
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyWindow(win)
        cv2.waitKey(1)
    return frame


def click_scale(frame: np.ndarray, ref_width_mm: float) -> float | None:
    """Click the two edges of a known-width reference; returns px/mm."""
    win = f"scale: click BOTH edges of the {ref_width_mm:g} mm reference (ESC = abort)"
    pts: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    try:
        while True:
            shown = frame.copy()
            for p in pts:
                cv2.circle(shown, p, 6, (0, 255, 0), 2)
            if len(pts) == 2:
                cv2.line(shown, pts[0], pts[1], (0, 255, 0), 2)
            cv2.imshow(win, shown)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                return None
            if len(pts) == 2:
                cv2.waitKey(400)
                break
    finally:
        cv2.destroyWindow(win)
        cv2.waitKey(1)

    px = float(np.hypot(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1]))
    if px < 5:
        print("clicked points nearly coincide; aborting scale", file=sys.stderr)
        return None
    ppm = px / ref_width_mm
    print(f"scale: {px:.1f} px over {ref_width_mm:g} mm -> {ppm:.3f} px/mm")
    return ppm


def show_image(path: Path) -> None:
    img = cv2.imread(str(path))
    if img is None:
        return
    win = f"{path.name}  [any key to continue]"
    cv2.imshow(win, img)
    cv2.waitKey(0)
    cv2.destroyWindow(win)
    cv2.waitKey(1)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze_capture(
    name: str, image_path: Path, run_dir: Path, args
) -> ContactEstimate:
    print(f"\n=== {name} -> {run_dir}")
    finger = None
    if args.finger_length is not None:
        finger = FingerGeometry(
            finger_length=args.finger_length, pad_length=args.L,
            pad_start=args.pad_start, tip_clearance=args.tip_clearance,
            palm_standoff=args.palm_standoff,
        )
    params = ContactParams(
        px_per_mm=args.px_per_mm, object_type=args.object_type,
        closing_axis=args.closing_axis, k_max=args.k_max, delta=args.delta,
        L=args.L, w_pad=args.w_pad, ds=args.ds, smoothing=args.smoothing,
        sweep_k=tuple(float(v) for v in args.sweep_k.split(",") if v.strip()),
        finger=finger,
    )
    est, summary, paths = analyze_image(
        image_path, run_dir, name, params, index_csv=args.out_root / "index.csv"
    )

    print(f"contact L/R : {est.left.contact_length:.2f} / "
          f"{est.right.contact_length:.2f} mm")
    print(f"total area  : {est.total_area:.2f} mm^2")
    print(f"fraction    : {est.mean_fraction:.4f}   "
          f"(sweep {summary['k_max_sweep_mean_fraction']})")
    print(f"figure      : {paths['contact_fig']}")
    return est


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--image", type=Path, default=None,
                    help="process this photo instead of using the camera")
    ap.add_argument("--name", default=None,
                    help="object name (default: image stem, or asked per capture)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--px-per-mm", type=float, default=None)
    ap.add_argument("--ref-width-mm", type=float, default=None,
                    help="known reference width; click its edges to set scale")
    ap.add_argument("--object-type", choices=["prismatic", "axisymmetric"],
                    default="prismatic")
    ap.add_argument("--closing-axis", choices=["x", "y"], default="x")
    ap.add_argument("--k-max", type=float, default=2.0)
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--L", type=float, default=4.0,
                    help="pad length; = FingerGeometry.pad_length when "
                         "--finger-length is set")
    ap.add_argument("--w-pad", type=float, default=12.0)
    ap.add_argument("--finger-length", type=float, default=None,
                    help="palm-to-tip length, mm; enables the drop-depth "
                         "model (pad placement constrained by geometry)")
    ap.add_argument("--pad-start", type=float, default=0.0,
                    help="fingertip to pad lower edge, mm")
    ap.add_argument("--tip-clearance", type=float, default=2.0,
                    help="min fingertip height above the table, mm")
    ap.add_argument("--palm-standoff", type=float, default=5.0,
                    help="palm clearance above the object top, mm")
    ap.add_argument("--ds", type=float, default=0.25)
    ap.add_argument("--smoothing", type=float, default=0.2)
    ap.add_argument("--sweep-k", default=SWEEP_K_DEFAULT,
                    help="comma list of k_max values logged in summary.json")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--no-show", action="store_true",
                    help="skip result windows (headless)")
    args = ap.parse_args()

    if args.px_per_mm is None and args.ref_width_mm is None:
        ap.error("scale required: pass --px-per-mm or --ref-width-mm")

    args.out_root.mkdir(parents=True, exist_ok=True)

    def run_one(frame: np.ndarray | None, image_src: Path | None) -> None:
        name = args.name or (image_src.stem if image_src else None)
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
            image_bgr = cv2.imread(str(image_src))
            if image_bgr is None:
                raise SystemExit(f"could not read {image_src}")
            image_path = run_dir / f"{name}{image_src.suffix}"
            cv2.imwrite(str(image_path), image_bgr)

        if args.px_per_mm is None:
            shown = frame if frame is not None else cv2.imread(str(image_path))
            ppm = click_scale(shown, args.ref_width_mm)
            if ppm is None:
                print("no scale set; run discarded", file=sys.stderr)
                return
            args.px_per_mm = ppm

        analyze_capture(name, image_path, run_dir, args)
        if not args.no_show:
            show_image(run_dir / f"{image_path.stem}_contact.png")

    if args.image is not None:
        run_one(None, args.image)
        return 0

    # camera session: capture objects until Q/ESC
    while True:
        frame = preview_and_capture(args.camera)
        if frame is None:
            break
        run_one(frame, None)
        args.name = None  # ask again for the next object
        print("\nback to preview - SPACE for next object, Q to finish")
        time.sleep(0.3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
