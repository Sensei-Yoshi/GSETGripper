"""Stage check for visual-semantic object description.

    python scripts/check_perception.py             # deterministic dry run
    python scripts/check_perception.py path.png    # describe a real image (needs GEMINI key unless dry_run)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.config import load_config  # noqa: E402
from modules.perception import describe  # noqa: E402


def main() -> int:
    cfg = load_config()

    if len(sys.argv) > 1:
        import cv2

        image = cv2.imread(sys.argv[1])  # live description (needs GEMINI_API_KEY)
        print("description:", describe(image, cfg))
        return 0

    cfg.models.dry_run = True
    print("description:", describe(None, cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
