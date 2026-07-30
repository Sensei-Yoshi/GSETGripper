"""Canonical catalog and factory for the six experiment strategies."""

from __future__ import annotations

from ..config import EXPERIMENT_IDS, Config, ExperimentMethod
from ..retrieval import RetrievalMode
from .e1 import E1Strategy
from .e2 import E2Strategy
from .e3 import E3Strategy
from .e4 import E4Strategy
from .e5 import E5Strategy
from .e6 import E6Strategy
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
        ExperimentMethod.MEASURED_VLM,
        "Measured-input zero-shot",
        "E1 plus authoritative mass, roughness, and projected contact.",
        1,
        uses_measurements=True,
        uses_roughness=True,
        uses_projected_contact=True,
    ),
    "e3": ExperimentSpec(
        "e3",
        ExperimentMethod.SEMANTIC_RETRIEVAL_VLM,
        "Semantic experiential retrieval",
        "Image plus semantic-only experiences for the active grippers; no sensor values exposed.",
        1,
        retrieval_mode=RetrievalMode.SEMANTIC_ONLY,
    ),
    "e4": ExperimentSpec(
        "e4",
        ExperimentMethod.HYBRID_RETRIEVAL_VLM,
        "Semantic + mass retrieval",
        "Semantic experiences ranked with semantic similarity and mass only.",
        1,
        uses_measurements=True,
        retrieval_mode=RetrievalMode.HYBRID,
    ),
    "e5": ExperimentSpec(
        "e5",
        ExperimentMethod.HYBRID_RETRIEVAL_VLM,
        "Semantic + mass + roughness retrieval",
        "E4 plus authoritative continuous roughness in ranking and VLM evidence.",
        1,
        uses_measurements=True,
        uses_roughness=True,
        roughness_representation="continuous",
        retrieval_mode=RetrievalMode.HYBRID,
    ),
    "e6": ExperimentSpec(
        "e6",
        ExperimentMethod.HYBRID_RETRIEVAL_VLM,
        "Full sensor-fusion retrieval",
        "E5 plus authoritative projected contact fraction in ranking and VLM evidence.",
        1,
        uses_measurements=True,
        uses_roughness=True,
        uses_projected_contact=True,
        roughness_representation="continuous",
        retrieval_mode=RetrievalMode.HYBRID,
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
    "e6": E6Strategy,
}


def create_strategy(cfg: Config, experiment_id: str) -> ExperimentStrategy:
    normalized = experiment_id.lower()
    definition = cfg.experiment(normalized)
    spec = EXPERIMENT_CATALOG[normalized]
    return STRATEGY_TYPES[normalized](cfg, spec, definition)


def experiment_display_name(experiment_id: str) -> str:
    spec = EXPERIMENT_CATALOG[experiment_id.lower()]
    return f"{spec.experiment_id.upper()} — {spec.label}"
