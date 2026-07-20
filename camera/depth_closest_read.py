import argparse
import json
import sys
import time
from pathlib import Path

CAPTURE_FILE_PATH = Path(__file__).parent / "captures" / "latest_capture.json"
DEFAULT_MAX_AGE_SECONDS = 30.0


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

    if args.raw:
        print(f"{depth_mm:.1f}")
    else:
        print(f"Closest depth: {depth_mm:.1f} mm (captured {age_seconds:.1f}s ago)")

    if age_seconds > args.max_age:
        print(
            f"Warning: capture is {age_seconds:.1f}s old, older than the {args.max_age:.1f}s freshness threshold.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
