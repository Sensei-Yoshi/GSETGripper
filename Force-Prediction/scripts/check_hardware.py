"""Stage check: hardware interface + staircase, on the mock bench (no hardware).

    python scripts/check_hardware.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.collect import measure_pair  # noqa: E402
from modules.config import load_config  # noqa: E402
from modules.contracts import Gripper  # noqa: E402
from modules.hardware import make_mock_bench, synthetic_objects  # noqa: E402


def main() -> int:
    cfg = load_config()
    bench, devices = make_mock_bench(cfg)
    coarse = cfg.collection.coarse_step_n
    fine = cfg.collection.fine_step_n
    for obj in synthetic_objects(cfg, 3):
        bench.set_object(obj)
        print(f"\n{obj.object_id}: mass={obj.mass_g:.0f}g rough={obj.roughness_class} "
              f"a={obj.projected_contact_fraction:.2f}")
        for gripper in (Gripper.GECKO, Gripper.SILICONE):
            bench.mounted_gripper = gripper
            feasible, force, trials = measure_pair(
                devices, cfg, cfg.collection.repeats, coarse, fine
            )
            print(f"  {gripper.value:8}: feasible={feasible} min_force={force} "
                  f"trials={trials} load_cell~{devices.load_cell.read_n():.2f}N")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
