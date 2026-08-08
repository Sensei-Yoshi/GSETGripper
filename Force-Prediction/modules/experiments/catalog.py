"""Canonical catalog and factory for the active experiment strategies."""

from __future__ import annotations

from ..config import EXPERIMENT_IDS, Config, ExperimentMethod
from ..retrieval import RetrievalMode
from .e1 import E1Strategy
from .e2 import E2Strategy
from .e3 import E3Strategy
from .e4 import E4Strategy
from .e5 import E5Strategy
from .helper import ExperimentSpec, ExperimentStrategy

EXPERIMENT_CATALOG: dict[str, ExperimentSpec] = {
    "e1": ExperimentSpec(
        "e1",
        ExperimentMethod.VISION_VLM,
        "Vision-only zero-shot",
        "Object image and fixed gripper context; no sensors or experiences.",
        1,
    ),
    "e2": ExperimentSpec(
        "e2",
        ExperimentMethod.SEMANTIC_RETRIEVAL_VLM,
        "Semantic experiential retrieval",
        "Image plus semantic-only experiences for the active grippers; no sensor values exposed.",
        1,
        retrieval_mode=RetrievalMode.SEMANTIC_ONLY,
    ),
    "e3": ExperimentSpec(
        "e3",
        ExperimentMethod.HYBRID_RETRIEVAL_VLM,
        "Semantic + mass retrieval",
        "Semantic experiences ranked with semantic similarity and mass only.",
        1,
        uses_measurements=True,
        retrieval_mode=RetrievalMode.HYBRID,
        ranking_features=("semantic", "mass"),
        visible_condition_fields=("mass_g",),
    ),
    "e4": ExperimentSpec(
        "e4",
        ExperimentMethod.HYBRID_RETRIEVAL_VLM,
        "Semantic + mass + roughness retrieval",
        "E3 plus authoritative continuous roughness in ranking and VLM evidence.",
        1,
        uses_measurements=True,
        uses_roughness=True,
        retrieval_mode=RetrievalMode.HYBRID,
        ranking_features=("semantic", "mass", "roughness"),
        visible_condition_fields=("mass_g", "roughness_index"),
    ),
    "e5": ExperimentSpec(
        "e5",
        ExperimentMethod.HYBRID_RETRIEVAL_VLM,
        "Contact-conditioned experience reasoning",
        "E4-ranked surfaces plus authoritative projected-contact condition evidence.",
        1,
        uses_measurements=True,
        uses_roughness=True,
        uses_projected_contact=True,
        retrieval_mode=RetrievalMode.HYBRID,
        ranking_features=("semantic", "mass", "roughness"),
        visible_condition_fields=(
            "mass_g",
            "roughness_index",
            "projected_contact_fraction",
        ),
    ),
}

if tuple(EXPERIMENT_CATALOG) != EXPERIMENT_IDS:
    raise RuntimeError("experiment catalog IDs do not match the validated config contract")

STRATEGY_TYPES: dict[str, type[ExperimentStrategy]] = {
    "e1": E1Strategy,
    "e2": E2Strategy,
    "e3": E3Strategy,
    "e4": E4Strategy,
    "e5": E5Strategy,
}


def create_strategy(cfg: Config, experiment_id: str) -> ExperimentStrategy:
    normalized = experiment_id.lower()
    definition = cfg.experiment(normalized)
    spec = EXPERIMENT_CATALOG[normalized]
    return STRATEGY_TYPES[normalized](cfg, spec, definition)


def experiment_display_name(experiment_id: str) -> str:
    spec = EXPERIMENT_CATALOG[experiment_id.lower()]
    return f"{spec.experiment_id.upper()} — {spec.label}"
