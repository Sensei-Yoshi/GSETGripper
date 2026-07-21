"""Stage check: hybrid retrieval + paired-row deltas (offline, mock embeddings).

    python scripts/check_retrieval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from force_prediction.config import load_config  # noqa: E402
from force_prediction.contracts import Gripper, Query, load_experiences  # noqa: E402
from force_prediction.hardware import fabricate_records  # noqa: E402
from force_prediction.retrieval import ExperienceIndex, build_embedding_text  # noqa: E402


def main() -> int:
    cfg = load_config()
    cfg.models.dry_run = True  # force mock embeddings for a self-contained run

    records = load_experiences(cfg.path("experiences")) or fabricate_records(cfg, 40)
    index = ExperienceIndex(cfg).fit(records)

    probe = records[0]
    query = Query(
        object_id="probe", image_path="", mass_g=probe.mass_g,
        roughness_class=probe.roughness_class,
        projected_contact_fraction=probe.projected_contact_fraction,
        semantic_description=probe.semantic_description,
    )
    qvec = index.provider.embed(build_embedding_text(
        query.semantic_description, query.mass_g, query.roughness_class,
        query.projected_contact_fraction, cfg))

    for gripper in (Gripper.GECKO, Gripper.SILICONE):
        print(f"\nTop-{cfg.retrieval.k} {gripper.value} for mass={query.mass_g:.0f}g "
              f"rough={query.roughness_class} a={query.projected_contact_fraction:.2f}:")
        for r in index.retrieve(query, qvec, gripper, exclude_object_id="probe"):
            print(f"  {r.record.object_id} score={r.score:.3f} "
                  f"force={r.record.min_force_n} other={r.other_gripper_min_force_n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
