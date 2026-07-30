"""Stage check: the full Gemini-backed paired-retrieval VLM pipeline (E6).

Prints each per-gripper prediction and the final deterministic selection.

    python scripts/check_pipeline.py path.png --confirm-gemini-cost
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.config import load_config  # noqa: E402
from modules.contracts import load_experiences  # noqa: E402
from modules.hardware import fabricate_records  # noqa: E402
from modules.pipeline import Pipeline, query_input_from_object  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Object image for the held-out pipeline query.")
    parser.add_argument("--confirm-gemini-cost", action="store_true")
    args = parser.parse_args()
    if not args.confirm_gemini_cost:
        parser.error("pipeline check requires --confirm-gemini-cost")
    cfg = load_config()

    records = load_experiences(cfg.path("experiences")) or fabricate_records(cfg, 40)
    held_out = records[0].object_id
    train = [r for r in records if r.object_id != held_out]
    test = [r for r in records if r.object_id == held_out]
    import cv2

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not decode object image: {args.image}")
    query = query_input_from_object(test, cfg)
    query.image_bgr = image

    pipe = Pipeline(cfg, "e6").fit(train)
    result = pipe.predict(query)

    print(f"held-out object: {held_out}")
    for g, p in result.candidate_predictions.items():
        print(f"  {g:8}: feasible={p.feasible} force={p.predicted_normal_force_n}N "
              f":: {p.reasoning_trace}")
    print(f"\nSELECTED: {result.desired_gripper} @ {result.predicted_normal_force_n}N")
    print(f"reason: {result.reasoning_trace}")
    for t in test:
        print(f"  truth {t.gripper.value}: feasible={t.feasible} min_force={t.min_force_n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
