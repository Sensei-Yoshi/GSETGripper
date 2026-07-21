import argparse
import json
import sys
import time
from pathlib import Path

CAPTURE_FILE_PATH = Path(__file__).parent / "captures" / "latest_capture.json"
DEFAULT_MAX_AGE_SECONDS = 30.0

# Fixed rig geometry (right triangle: camera -> horizontal offset x -> object top,
# hypotenuse d = sensor depth reading). h = sqrt(d^2 - x^2) is how far the object's
# top sits below the camera; object height = total camera mount height - h.
CAMERA_HEIGHT_MM = 1193.8  # 3 ft 11 in (47 in) camera mount height above ground
CAMERA_HORIZONTAL_DISTANCE_MM = 444.5  # 17.5 in horizontal camera-to-object distance x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the most recently captured closest-depth value from depth_closest_viewer.py."
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print only the depth value in mm, with no extra text.",
    )
    parser.add_argument(
        "--max-age",
        type=float,
        default=DEFAULT_MAX_AGE_SECONDS,
        help=f"Warn (and exit 2) if the capture is older than this many seconds. Default: {DEFAULT_MAX_AGE_SECONDS}.",
    )
    return parser.parse_args()


def compute_object_height_mm(slant_distance_mm: float) -> float | None:
    """Return object height above ground, or None if the geometry is invalid
    (slant distance d shorter than the fixed horizontal distance x)."""
    if slant_distance_mm < CAMERA_HORIZONTAL_DISTANCE_MM:
        return None
    height_below_camera_mm = (slant_distance_mm**2 - CAMERA_HORIZONTAL_DISTANCE_MM**2) ** 0.5
    return CAMERA_HEIGHT_MM - height_below_camera_mm


def main() -> int:
    args = parse_args()

    if not CAPTURE_FILE_PATH.exists():
        print(f"No capture found at {CAPTURE_FILE_PATH}. Run depth_closest_viewer.py and press 'c' first.", file=sys.stderr)
        return 1

    try:
        payload = json.loads(CAPTURE_FILE_PATH.read_text())
        depth_mm = float(payload["depth_mm"])
        captured_at_epoch = float(payload["captured_at_epoch"])
    except (json.JSONDecodeError, KeyError, ValueError) as error:
        print(f"Capture file at {CAPTURE_FILE_PATH} is malformed: {error}", file=sys.stderr)
        return 1

    age_seconds = time.time() - captured_at_epoch

    object_height_mm = compute_object_height_mm(depth_mm)

    if args.raw:
        if object_height_mm is None:
            print(f"{depth_mm:.1f} nan")
        else:
            print(f"{depth_mm:.1f} {object_height_mm:.1f}")
    else:
        print(f"Closest depth: {depth_mm:.1f} mm (captured {age_seconds:.1f}s ago)")
        if object_height_mm is None:
            print(
                f"Object height: undefined (depth {depth_mm:.1f} mm is shorter than the "
                f"fixed horizontal distance {CAMERA_HORIZONTAL_DISTANCE_MM:.1f} mm)",
                file=sys.stderr,
            )
        else:
            print(f"Object height: {object_height_mm:.1f} mm")

    if age_seconds > args.max_age:
        print(
            f"Warning: capture is {age_seconds:.1f}s old, older than the {args.max_age:.1f}s freshness threshold.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
