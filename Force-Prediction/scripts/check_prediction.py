"""Stage check: one joint E2 VLM response for both grippers.

    python scripts/check_prediction.py path.png  # one Gemini call (or cache hit)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.config import load_config  # noqa: E402
from modules.contracts import Query  # noqa: E402
from modules.prediction import vlm_predict_joint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Object image for the E2 prediction check.")
    args = parser.parse_args()
    import cv2

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not decode object image: {args.image}")
    cfg = load_config().model_copy(deep=True)
    query = Query(
        object_id="probe",
        image_path="",
        mass_g=420.0,
        roughness_class=2,
        projected_contact_fraction=0.83,
        semantic_description="smooth rigid plastic bottle",
    )
    prediction = vlm_predict_joint(
        cfg,
        query,
        image,
        [],
        instruction=cfg.prompts.experiments["e2"],
        include_measured=True,
        include_retrieval=False,
    )
    for gripper, result in (("gecko", prediction.gecko), ("silicone", prediction.silicone)):
        print(
            f"{gripper:8}: pred={result.predicted_normal_force_n}N "
            f"feasible={result.feasible} :: {result.reasoning_trace}"
        )
    print(f"model recommendation: {prediction.recommended_gripper}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
