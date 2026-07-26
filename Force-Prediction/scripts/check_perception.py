"""Stage check: descriptor + projected-contact-fraction proxy.

    python scripts/check_perception.py            # synthetic depth
    python scripts/check_perception.py path.png    # describe a real image (needs GEMINI key unless dry_run)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.config import load_config  # noqa: E402
from modules.hardware import make_mock_bench, synthetic_objects  # noqa: E402
from modules.perception import describe, projected_contact_fraction  # noqa: E402


def main() -> int:
    cfg = load_config()

    if len(sys.argv) > 1:
        import cv2

        image = cv2.imread(sys.argv[1])  # live description (needs GEMINI_API_KEY)
        print("description:", describe(image, cfg))
        return 0

    cfg.models.dry_run = True  # synthetic branch is fully offline
    bench, devices = make_mock_bench(cfg)
    for obj in synthetic_objects(cfg, 3):
        bench.set_object(obj)
        depth = devices.camera.capture_depth_mm()
        a = projected_contact_fraction(depth, cfg)
        print(f"{obj.object_id}: true a={obj.projected_contact_fraction:.2f} "
              f"recovered a={a:.2f}  desc={describe(devices.camera.capture_rgb(), cfg).description!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
