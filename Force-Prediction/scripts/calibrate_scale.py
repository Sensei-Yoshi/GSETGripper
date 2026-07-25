#!/usr/bin/env python3
"""Measure object dimensions with the camera scale from config.yaml.

The contact model works in millimetres, and the pixels-per-millimetre scale
lives in `config.yaml` under `geometry.px_per_mm`. This tool uses that value
to report a segmented object's real height/width, and lets you re-measure the
scale (click a known reference) — writing the new value straight back into
config.yaml.

A single camera's px/mm is only valid at ONE depth plane, so this is valid
only if every object's front face sits at the SAME camera distance as the
reference the config value was measured from. Recommended rig: camera flat on
the table looking horizontally (not tilted), a tape line marking where object
fronts go, objects centred.

Usage:
    VENV=/Users/premshah/Desktop/Robotics/GSET/env/bin/python
    $VENV scripts/calibrate_scale.py                 # camera 0
    $VENV scripts/calibrate_scale.py --camera 1

Keys (in the OpenCV window):
    SPACE / c   Freeze the live frame (or unfreeze back to live).
    s           Segment the object (background remover) and report its
                height/width in px and in mm (using config geometry.px_per_mm).
    click x2    Click two points of known real separation to re-measure scale.
    d           Enter that real distance (mm); measures px/mm and writes it to
                geometry.px_per_mm in config.yaml automatically.
    r           Clear the clicked points.
    q / ESC     Quit.

Reuses open_camera() from collect_images.py, the background remover from
scripts/contact_model/extract_object_outline.py, and the scale from config.yaml.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# scripts/ is on sys.path[0] when run directly -> reuse the camera opener.
from collect_images import open_camera

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))  # make force_prediction importable
_CONTACT_MODEL_DIR = Path(__file__).resolve().parent / "contact_model"
sys.path.insert(0, str(_CONTACT_MODEL_DIR))

from force_prediction.config import load_config  # noqa: E402

WINDOW = "scale measurement"


# --------------------------------------------------------------------------- #
# Segmentation (reuses the contact-model background remover)
# --------------------------------------------------------------------------- #
def _load_session():
    import extract_object_outline as outline
    from rembg import new_session

    print(f"Loading background-removal model: {outline.REMBG_MODEL}")
    return new_session(outline.REMBG_MODEL)


def segment_mask(frame_bgr: np.ndarray, session) -> np.ndarray:
    """Return a clean 0/255 mask of the largest foreground object."""
    import extract_object_outline as outline
    from rembg import remove

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgba = np.asarray(remove(rgb, session=session))
    return outline.largest_foreground_component(rgba[:, :, 3])


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(x, y, w, h) bounding box of the mask, or None if empty."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


# --------------------------------------------------------------------------- #
# Config write-back (surgical: replaces only the number, keeps all comments)
# --------------------------------------------------------------------------- #
_CONFIG_PATH = _REPO / "config.yaml"
_PX_LINE = re.compile(r"^(\s*px_per_mm:\s*)([-+0-9.eE]+)(.*)$", re.MULTILINE)


def write_config_px_per_mm(value: float) -> float:
    """Rewrite geometry.px_per_mm in config.yaml in place; return written value.

    Only the numeric token is replaced, so indentation and inline comments are
    preserved. The edited text is parsed with a fresh yaml load and checked to
    hold the new value *before* it is written, so a bad edit never lands on
    disk. (load_config is lru_cached, so it cannot be used to verify a change
    within a running process.)
    """
    written = float(f"{value:.4f}")
    text = _CONFIG_PATH.read_text()
    if not _PX_LINE.search(text):
        raise RuntimeError("px_per_mm key not found in config.yaml")
    new_text = _PX_LINE.sub(
        lambda m: f"{m.group(1)}{written:.4f}{m.group(3)}", text, count=1)

    got = (yaml.safe_load(new_text) or {}).get("geometry", {}).get("px_per_mm")
    if got is None or abs(float(got) - written) > 1e-9:
        raise RuntimeError("edited config.yaml did not parse to the new value")
    _CONFIG_PATH.write_text(new_text)
    return written


# --------------------------------------------------------------------------- #
# Interactive state
# --------------------------------------------------------------------------- #
class Calibrator:
    def __init__(self, px_per_mm: float) -> None:
        self.px_per_mm = px_per_mm                 # from config.yaml
        self.points: list[tuple[int, int]] = []
        self.measured_px_per_mm: float | None = None  # from click+distance
        self.seg_overlay: np.ndarray | None = None
        self.seg_dims: tuple[int, int] | None = None  # (w_px, h_px)

    # -- mouse ------------------------------------------------------------- #
    def on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 2:
            self.points.append((x, y))

    # -- re-measure scale (verification only; does not change segmentation) - #
    def enter_distance(self) -> None:
        if len(self.points) != 2:
            print("Click exactly two points first.")
            return
        px = float(np.hypot(
            self.points[0][0] - self.points[1][0],
            self.points[0][1] - self.points[1][1],
        ))
        raw = input("Real distance between the two points, in mm: ").strip()
        try:
            mm = float(raw)
        except ValueError:
            print(f"  '{raw}' is not a number; ignored.")
            return
        if mm <= 0 or px < 1:
            print("  need a positive distance and non-coincident points.")
            return
        self.measured_px_per_mm = px / mm
        diff = 100.0 * (self.measured_px_per_mm - self.px_per_mm) / self.px_per_mm
        old = self.px_per_mm
        try:
            written = write_config_px_per_mm(self.measured_px_per_mm)
        except Exception as exc:  # keep running even if the write fails
            print(f"  measured px_per_mm = {self.measured_px_per_mm:.4f} "
                  f"({diff:+.1f}% vs config); could NOT update config.yaml: {exc}")
            return
        self.px_per_mm = written
        print(f"  measured px_per_mm = {self.measured_px_per_mm:.4f} "
              f"({diff:+.1f}% vs previous)")
        print(f"  updated config.yaml geometry.px_per_mm: "
              f"{old:.4f} -> {written:.4f}")

    # -- segmentation (uses the config scale) ------------------------------ #
    def segment(self, frame: np.ndarray, session) -> None:
        mask = segment_mask(frame, session)
        box = mask_bbox(mask)
        overlay = frame.copy()
        if box is None:
            print("No foreground object found.")
            self.seg_overlay = overlay
            self.seg_dims = None
            return
        x, y, w, h = box
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 165, 255), 2)
        self.seg_overlay = overlay
        self.seg_dims = (w, h)
        print(f"segmented object: width {w} px, height {h} px  ->  "
              f"width {w / self.px_per_mm:.1f} mm, "
              f"height {h / self.px_per_mm:.1f} mm  "
              f"(px_per_mm={self.px_per_mm:.4f} from config)")

    # -- rendering --------------------------------------------------------- #
    def render(self, base: np.ndarray, live: bool) -> np.ndarray:
        img = (self.seg_overlay if self.seg_overlay is not None else base).copy()
        for pt in self.points:
            cv2.circle(img, pt, 5, (0, 255, 255), -1)
        if len(self.points) == 2:
            cv2.line(img, self.points[0], self.points[1], (0, 255, 255), 2)

        lines = [
            "LIVE  |  SPACE/c: freeze" if live
            else "FROZEN  |  s: segment, click 2 pts + d: measure->config, "
                 "r: reset, SPACE/c: live",
            f"px_per_mm (config): {self.px_per_mm:.4f}",
        ]
        if self.measured_px_per_mm:
            lines.append(f"measured: {self.measured_px_per_mm:.4f}")
        if self.seg_dims:
            w, h = self.seg_dims
            lines.append(f"object: {w / self.px_per_mm:.1f} x "
                         f"{h / self.px_per_mm:.1f} mm")
        y = 24
        for text in lines:
            cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)
            y += 26
        return img


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="camera index")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    px_per_mm = load_config().geometry.px_per_mm
    print(f"Using px_per_mm = {px_per_mm:.4f} from config.yaml (geometry.px_per_mm)")

    try:
        cap = open_camera(args.camera, args.width, args.height)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    cal = Calibrator(px_per_mm)
    session = None
    frozen: np.ndarray | None = None
    live = True

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, cal.on_mouse)
    print("Ready. SPACE/c to freeze a frame, then 's' to segment and measure.")

    try:
        while True:
            if live:
                ok, frame = cap.read()
                if not ok:
                    print("Camera frame read failed.", file=sys.stderr)
                    return 1
                base = frame
            else:
                base = frozen

            cv2.imshow(WINDOW, cal.render(base, live))
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key in (ord(" "), ord("c")):
                live = not live
                if not live:
                    frozen = frame.copy()
                    cal.points.clear()
                    cal.seg_overlay = None
                    cal.seg_dims = None
                else:
                    cal.seg_overlay = None
            elif not live and key == ord("s"):
                if session is None:
                    session = _load_session()
                cal.segment(frozen, session)
            elif not live and key == ord("d"):
                cal.enter_distance()
            elif not live and key == ord("r"):
                cal.points.clear()
                cal.seg_overlay = None
                cal.seg_dims = None
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
