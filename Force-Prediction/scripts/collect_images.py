#!/usr/bin/env python3
"""Interactively save camera frames for the real chosen-objects dataset.

Usage:
    python scripts/collect_images.py
    python scripts/collect_images.py --camera 1 --width 1920 --height 1080

Keys:
    SPACE  Save the current frame.
    q/ESC  Quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

DEFAULT_OUTPUT_DIR = Path(
    "/Users/premshah/Desktop/Robotics/GSET/GSETGripper/Force-Prediction/"
    "data/real_chosen_objects"
)
WINDOW_NAME = "Data Collection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")
    parser.add_argument("--width", type=int, default=1280, help="Requested frame width.")
    parser.add_argument("--height", type=int, default=720, help="Requested frame height.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory in which to save images (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def next_image_number(output_dir: Path) -> int:
    """Return the next unused number among image_0001.png-style files."""
    numbers: list[int] = []
    for path in output_dir.glob("image_*.png"):
        try:
            numbers.append(int(path.stem.removeprefix("image_")))
        except ValueError:
            continue
    return max(numbers, default=0) + 1


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # The first few reads can fail while the camera or macOS permission prompt
    # initializes, so retry briefly before reporting an error.
    if cap.isOpened():
        for _ in range(100):
            ok, _ = cap.read()
            if ok:
                return cap
            time.sleep(0.1)

    cap.release()
    raise RuntimeError(
        f"Could not read from camera {camera_index}. Check that no other app is "
        "using it and that your terminal has camera permission in System Settings "
        "> Privacy & Security > Camera."
    )


def main() -> int:
    args = parse_args()
    args.output_dir.expanduser().mkdir(parents=True, exist_ok=True)
    output_dir = args.output_dir.expanduser().resolve()

    try:
        cap = open_camera(args.camera, args.width, args.height)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    image_number = next_image_number(output_dir)
    print(f"Saving images to: {output_dir}")
    print("Press SPACE to save an image; press q or ESC to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame read failed.", file=sys.stderr)
                return 1

            # Draw instructions only on the preview. The saved frame remains clean.
            preview = frame.copy()
            cv2.putText(
                preview,
                "SPACE: save image    q/ESC: quit",
                (12, preview.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                image_path = output_dir / f"image_{image_number:04d}.png"
                if not cv2.imwrite(str(image_path), frame):
                    print(f"Failed to save {image_path}", file=sys.stderr)
                    return 1
                print(f"Saved {image_path}")
                image_number += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
