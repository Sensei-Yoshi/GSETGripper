"""Stage check: one per-gripper VLM structured prediction.

    python scripts/check_prediction.py            # dry-run stub, no network
    python scripts/check_prediction.py --live      # real Gemini call (needs GEMINI_API_KEY)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from force_prediction.config import load_config  # noqa: E402
from force_prediction.contracts import CandidateQuery, Gripper  # noqa: E402
from force_prediction.physics import PhysicsModel, PhysicsParams  # noqa: E402
from force_prediction.prediction import vlm_predict_gripper  # noqa: E402


def main() -> int:
    cfg = load_config()
    cfg.models.dry_run = "--live" not in sys.argv

    physics = PhysicsModel(PhysicsParams.from_config(cfg), cfg)
    for gripper in (Gripper.GECKO, Gripper.SILICONE):
        cq = CandidateQuery(
            object_id="probe", image_path="", mass_g=420.0, roughness_class=2,
            projected_contact_fraction=0.83, semantic_description="smooth rigid plastic bottle",
            candidate_gripper=gripper,
        )
        est = physics.min_force(gripper, cq.mass_g, cq.roughness_class, cq.projected_contact_fraction)
        pred = vlm_predict_gripper(cfg, cq, None, [], est, include_paired=True)
        print(f"{gripper.value:8}: physics={est.min_force_n} -> pred={pred.predicted_normal_force_n}N "
              f"feasible={pred.feasible} :: {pred.reasoning_trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
