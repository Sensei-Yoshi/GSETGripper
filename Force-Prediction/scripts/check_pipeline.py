"""Stage check: the paired-retrieval VLM pipeline (E4), offline.

Prints every stage: per-gripper physics estimate, prediction, and the final
deterministic selection.

    python scripts/check_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.config import load_config  # noqa: E402
from modules.contracts import load_experiences  # noqa: E402
from modules.hardware import fabricate_records  # noqa: E402
from modules.pipeline import Pipeline, query_input_from_object  # noqa: E402


def main() -> int:
    cfg = load_config()
    cfg.models.dry_run = True  # self-contained, no network

    records = load_experiences(cfg.path("experiences")) or fabricate_records(cfg, 40)
    held_out = records[0].object_id
    train = [r for r in records if r.object_id != held_out]
    test = [r for r in records if r.object_id == held_out]

    pipe = Pipeline(cfg, "e4").fit(train)
    result = pipe.predict(query_input_from_object(test, cfg))

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
