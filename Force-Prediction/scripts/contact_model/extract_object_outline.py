"""Remove an image background and fit a smooth curve to the object outline.

Edit ``IMAGE_PATH`` below, then run:

    python scripts/contact_model/extract_object_outline.py

The first run may download the rembg ISNet model. Results are written beside
the input image unless ``OUTPUT_DIR`` is changed.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import splev, splprep

try:
    from rembg import new_session, remove
except ImportError as exc:
    raise SystemExit(
        "rembg is required. Install the project dependencies with "
        "`python -m pip install -r requirements.txt`."
    ) from exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_PATH = Path(
    "/Users/premshah/Desktop/Robotics/GSET/GSETGripper/Force-Prediction/"
    "data/MatForce/plastic_cup.png"
)

# Set to a Path to save elsewhere. None saves into an ``outline_outputs``
# directory next to IMAGE_PATH.
OUTPUT_DIR: Path | None = None

REMBG_MODEL = "isnet-general-use"
ALPHA_THRESHOLD = 120
SPLINE_SMOOTHING_PIXELS = 2.0
SPLINE_INPUT_POINTS = 400
SPLINE_OUTPUT_POINTS = 1_000


def largest_foreground_component(alpha: np.ndarray) -> np.ndarray:
    """Convert a soft alpha matte into one clean mask for the main object."""
    binary = np.where(alpha >= ALPHA_THRESHOLD, 255, 0).astype(np.uint8)

    # Close small edge gaps without noticeably changing the silhouette.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        raise RuntimeError("rembg did not find a foreground object")

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    # Fill interior holes: only the outer silhouette is needed here.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def external_contour(mask: np.ndarray) -> np.ndarray:
    """Return the longest external contour as an (N, 2) float array."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("could not trace a contour from the foreground mask")
    contour = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    if len(contour) < 4:
        raise RuntimeError("the detected object contour is too small for a spline")
    return contour


def resample_closed_contour(points: np.ndarray, sample_count: int) -> np.ndarray:
    """Sample a closed contour at equal arc-length intervals."""
    closed = np.vstack((points, points[0]))
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = cumulative[-1]
    if total_length <= 0:
        raise RuntimeError("the detected object contour has zero length")

    distances = np.linspace(0.0, total_length, sample_count, endpoint=False)
    x = np.interp(distances, cumulative, closed[:, 0])
    y = np.interp(distances, cumulative, closed[:, 1])
    return np.column_stack((x, y))


def fit_closed_spline(contour: np.ndarray) -> np.ndarray:
    """Fit and densely sample a periodic cubic B-spline around the contour."""
    input_count = min(SPLINE_INPUT_POINTS, len(contour))
    uniform = resample_closed_contour(contour, input_count)
    closed_uniform = np.vstack((uniform, uniform[0]))

    # scipy's s is a total squared-error budget. Expressing the setting in
    # pixels makes the top-level tuning value easier to reason about.
    smoothing = len(closed_uniform) * SPLINE_SMOOTHING_PIXELS**2
    knots, _ = splprep(
        [closed_uniform[:, 0], closed_uniform[:, 1]],
        s=smoothing,
        per=True,
        k=min(3, len(closed_uniform) - 1),
    )
    parameter = np.linspace(0.0, 1.0, SPLINE_OUTPUT_POINTS, endpoint=False)
    x, y = splev(parameter, knots)
    return np.column_stack((x, y))


def write_svg(path: Path, curve: np.ndarray, width: int, height: int) -> None:
    """Write the sampled closed spline as a transparent vector SVG."""
    commands = [f"M {curve[0, 0]:.2f} {curve[0, 1]:.2f}"]
    commands.extend(f"L {x:.2f} {y:.2f}" for x, y in curve[1:])
    commands.append("Z")
    path_data = " ".join(commands)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">\n'
        f'  <path d="{path_data}" fill="none" stroke="#00ff66" '
        'stroke-width="3" stroke-linejoin="round"/>\n'
        "</svg>\n"
    )
    path.write_text(svg, encoding="utf-8")


def process_image(image_path: Path, output_dir: Path, session=None) -> list[Path]:
    """Run rembg, fit the outline spline, and save all result artifacts.

    Pass a pre-built ``session`` (from ``rembg.new_session``) to skip the
    per-call model load; callers such as the Streamlit page cache it.
    """
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"could not read image: {image_path}")

    if session is None:
        print(f"Loading rembg model: {REMBG_MODEL}")
        session = new_session(REMBG_MODEL)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgba = np.asarray(remove(image_rgb, session=session))
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise RuntimeError(f"rembg returned an unexpected image shape: {rgba.shape}")

    mask = largest_foreground_component(rgba[:, :, 3])
    contour = external_contour(mask)
    curve = fit_closed_spline(contour)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    cutout_path = output_dir / f"{stem}_cutout.png"
    mask_path = output_dir / f"{stem}_mask.png"
    overlay_path = output_dir / f"{stem}_spline_overlay.png"
    csv_path = output_dir / f"{stem}_spline_points.csv"
    svg_path = output_dir / f"{stem}_spline.svg"

    # Use the cleaned mask as alpha so the cutout and fitted outline agree.
    cutout_bgra = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    cutout_bgra[:, :, 3] = mask

    overlay = image_bgr.copy()
    integer_curve = np.rint(curve).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(overlay, [integer_curve], isClosed=True, color=(0, 255, 0), thickness=3)

    if not cv2.imwrite(str(cutout_path), cutout_bgra):
        raise OSError(f"could not write {cutout_path}")
    if not cv2.imwrite(str(mask_path), mask):
        raise OSError(f"could not write {mask_path}")
    if not cv2.imwrite(str(overlay_path), overlay):
        raise OSError(f"could not write {overlay_path}")

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("x", "y"))
        writer.writerows((f"{x:.4f}", f"{y:.4f}") for x, y in curve)

    height, width = image_bgr.shape[:2]
    write_svg(svg_path, curve, width, height)
    return [cutout_path, mask_path, overlay_path, csv_path, svg_path]


def main() -> int:
    image_path = IMAGE_PATH.expanduser().resolve()
    output_dir = (
        OUTPUT_DIR.expanduser().resolve()
        if OUTPUT_DIR is not None
        else image_path.parent / "outline_outputs"
    )

    try:
        outputs = process_image(image_path, output_dir)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Saved:")
    for output in outputs:
        print(f"  {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
