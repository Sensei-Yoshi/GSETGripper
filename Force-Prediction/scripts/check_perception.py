"""Stage check for visual-semantic object description.

    python scripts/check_perception.py path.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.config import load_config  # noqa: E402
from modules.perception import describe  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Object image to describe with Gemini.")
    args = parser.parse_args()
    cfg = load_config()
    import cv2

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not decode object image: {args.image}")
    print("description:", describe(image, cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
