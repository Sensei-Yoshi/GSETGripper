"""Stage check: physics calibration + minimum-force solver / feasibility.

    python scripts/check_physics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.config import load_config  # noqa: E402
from modules.contracts import Gripper, load_experiences  # noqa: E402
from modules.hardware import fabricate_records  # noqa: E402
from modules.physics import PhysicsModel, PhysicsParams, calibrate  # noqa: E402


def main() -> int:
    cfg = load_config()
    records = load_experiences(cfg.path("experiences")) or fabricate_records(cfg, 60)

    defaults = PhysicsModel(PhysicsParams.from_config(cfg), cfg)
    fitted = PhysicsModel(calibrate(records, cfg), cfg)
    print("calibrated params:", fitted.p)

    print("\nmin force by roughness index (mass=300g, a=0.8):")
    print(f"{'index':>7} {'sil_default':>12} {'sil_fit':>10} {'geo_default':>12} {'geo_fit':>10}")
    for roughness in (0.0, 250.0, 500.0, 750.0, 1000.0):
        sd = defaults.min_force(Gripper.SILICONE, 300, roughness, 0.8)
        sf = fitted.min_force(Gripper.SILICONE, 300, roughness, 0.8)
        gd = defaults.min_force(Gripper.GECKO, 300, roughness, 0.8)
        gf = fitted.min_force(Gripper.GECKO, 300, roughness, 0.8)
        print(f"{roughness:>7.1f} {str(sd.min_force_n):>12} {str(sf.min_force_n):>10} "
              f"{str(gd.min_force_n):>12} {str(gf.min_force_n):>10}")

    heavy = fitted.min_force(Gripper.GECKO, 1400, 1000.0, 0.4)
    print(f"\nheavy+rough gecko feasible={heavy.feasible} (expect infeasible near limit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
