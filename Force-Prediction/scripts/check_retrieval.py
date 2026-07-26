"""Stage check: E4 hybrid paired-object retrieval (offline mock embeddings)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.config import load_config  # noqa: E402
from modules.contracts import Query, load_experiences  # noqa: E402
from modules.hardware import fabricate_records  # noqa: E402
from modules.retrieval import ExperienceIndex  # noqa: E402


def main() -> int:
    cfg = load_config().model_copy(deep=True)
    cfg.models.dry_run = True
    records = load_experiences(cfg.path("experiences")) or fabricate_records(cfg, 40)
    index = ExperienceIndex(cfg).fit(records)
    probe = records[0]
    query = Query(
        object_id=probe.object_id,
        image_path="",
        mass_g=probe.mass_g,
        roughness_class=probe.roughness_class,
        projected_contact_fraction=probe.projected_contact_fraction,
        semantic_description=probe.semantic_description,
    )
    results = index.retrieve_objects(
        query,
        index.embed_query(query),
        exclude_object_id=query.object_id,
    )
    print(
        f"Top-{cfg.retrieval.k} paired objects for mass={query.mass_g:.0f}g "
        f"rough={query.roughness_class} a={query.projected_contact_fraction:.2f}:"
    )
    for result in results:
        print(
            f"  {result.object_id} score={result.score:.3f} "
            f"gecko={result.gecko_min_force_n} silicone={result.silicone_min_force_n}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
