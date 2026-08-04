"""Deterministic test-record factories with no production mock-hardware dependency."""

from __future__ import annotations

import numpy as np

from modules.config import Config
from modules.contracts import ExperienceRecord, Gripper, Meta


def fabricate_records(
    cfg: Config,
    n: int,
    seed: int | None = None,
) -> list[ExperienceRecord]:
    """Build varied paired records for unit tests and fold/retrieval checks."""
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    records: list[ExperienceRecord] = []
    scale = cfg.roughness.characteristic_scale
    for index in range(n):
        mass_g = float(np.exp(rng.uniform(np.log(20), np.log(1500))))
        roughness = float(rng.uniform(0.0, 4.0 * scale))
        contact = float(rng.uniform(0.3, 1.0))
        object_id = f"object_{index:03d}"
        for gripper, efficiency in (
            (Gripper.GECKO, 2.35),
            (Gripper.SILICONE, 2.0),
        ):
            roughness_factor = (
                1.0 + 0.10 * roughness / scale
                if gripper is Gripper.GECKO
                else 1.0 + 0.04 * roughness / scale
            )
            force = (mass_g / 1000.0 * 9.81) * roughness_factor / (efficiency * contact)
            feasible = force <= cfg.force.limit_n
            records.append(
                ExperienceRecord(
                    object_id=object_id,
                    image_path="",
                    mass_g=round(mass_g, 1),
                    roughness_index=round(roughness, 2),
                    projected_contact_fraction=round(contact, 3),
                    gripper=gripper,
                    min_force_n=round(force, 6) if feasible else None,
                    feasible=feasible,
                    failed_at_limit_n=None if feasible else cfg.force.limit_n,
                    semantic_description=f"synthetic test surface {index % 7}",
                    meta=Meta(n_trials=0, pad_id="test-factory"),
                )
            )
    return records
