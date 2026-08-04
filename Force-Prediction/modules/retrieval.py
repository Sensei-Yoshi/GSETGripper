"""Grouped physical-surface retrieval over semantics and measurement conditions.

E3 ranks by semantic cosine only; E4-E6 use nested subsets of the configured hybrid score. All return
the available gripper labels for every neighbor; request payloads expose only active ones.
Exact search is appropriate for this dataset (<1k objects), so no vector database
is required. The hybrid score is only a neighbor-ranking heuristic: it never
evaluates holding-force equations and never produces a force prediction.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, Field

from .config import Config
from .contracts import (
    ExperienceRecord,
    Gripper,
    ObjectRecord,
    Query,
    group_by_object,
)
from .models.gemini import get_client

RankingFeature = Literal["semantic", "mass", "roughness"]
MeasurementField = Literal[
    "mass_g",
    "roughness_index",
    "projected_contact_fraction",
]
ConditionRole = Literal["baseline", "controlled_variant"]

MEASUREMENT_FIELDS: tuple[MeasurementField, ...] = (
    "mass_g",
    "roughness_index",
    "projected_contact_fraction",
)


# --------------------------------------------------------------------------- #
# Embedding providers
# --------------------------------------------------------------------------- #
@runtime_checkable
class EmbeddingProvider(Protocol):
    def embed(
        self, text: str, image_bgr: np.ndarray | None = None, is_query: bool = False
    ) -> np.ndarray: ...


class GeminiEmbeddingProvider:
    """gemini-embedding-2: asymmetric retrieval formatting (Google's documented
    templates). Stored experiences are embedded as reference text, the query with the
    retrieval-query template, which aligns query<->corpus better than embedding
    both identically (SEMANTIC_SIMILARITY is explicitly not for retrieval)."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def embed(
        self, text: str, image_bgr: np.ndarray | None = None, is_query: bool = False
    ) -> np.ndarray:
        formatted = (
            f"task: search result | query: {text}"
            if is_query
            else f"title: none | text: {text}"
        )
        return get_client(self.cfg).embed(text=formatted, image_bgr=image_bgr)


def get_embedding_provider(cfg: Config) -> EmbeddingProvider:
    return GeminiEmbeddingProvider(cfg)


# --------------------------------------------------------------------------- #
# Embedding text + similarity terms
# --------------------------------------------------------------------------- #
def build_embedding_text(description: str) -> str:
    """Return semantic-only text for the vector embedding.

    Measured properties are scored explicitly below; including them in the vector
    as well would make the dashboard's retrieval weights impossible to interpret.
    """
    return description.strip()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def s_mass(m_q: float, m_i: float, sigma: float) -> float:
    return math.exp(-abs(math.log(m_q) - math.log(m_i)) / sigma)


def s_roughness(r_q: float, r_i: float, cfg: Config) -> float:
    """Continuous similarity for the recorded index; always in ``(0, 1]``."""
    return math.exp(-abs(r_q - r_i) / cfg.roughness.characteristic_scale)


def default_ranking_features(cfg: Config) -> tuple[RankingFeature, ...]:
    features: list[RankingFeature] = ["semantic", "mass"]
    if cfg.inputs.use_roughness:
        features.append("roughness")
    return tuple(features)


def normalized_weights(
    cfg: Config,
    ranking_features: tuple[RankingFeature, ...] | None = None,
) -> dict[str, float]:
    raw = cfg.retrieval.weights
    active = set(ranking_features or default_ranking_features(cfg))
    values = {
        "semantic": raw.semantic if "semantic" in active else 0.0,
        "mass": raw.mass if "mass" in active else 0.0,
        "roughness": raw.roughness if "roughness" in active else 0.0,
    }
    total = sum(values.values())
    if total <= 0:
        raise ValueError("at least one retrieval weight must be positive")
    return {name: value / total for name, value in values.items()}


# --------------------------------------------------------------------------- #
# Retrieved experience (payload-ready)
# --------------------------------------------------------------------------- #
class RetrievalMode(StrEnum):
    SEMANTIC_ONLY = "semantic_only"
    HYBRID = "hybrid"


class SimilarityBreakdown(BaseModel):
    mode: RetrievalMode = RetrievalMode.HYBRID
    semantic: float = Field(ge=-1.0, le=1.0)
    mass: float | None = Field(default=None, ge=0.0, le=1.0)
    roughness: float | None = Field(default=None, ge=0.0, le=1.0)
    semantic_contribution: float
    mass_contribution: float | None = None
    roughness_contribution: float | None = None
    total: float


class ConditionComparison(BaseModel):
    """Experiment-visible changes from a physical surface's baseline condition."""

    reference_condition_id: str = "baseline"
    changed_fields: tuple[MeasurementField, ...]
    unchanged_fields: tuple[MeasurementField, ...]
    deltas: dict[MeasurementField, float]
    # Signed change in each gripper's measured min force from this surface's
    # baseline to the variant (variant - baseline). This is the *within-surface*
    # causal effect of the changed field(s); the model must apply it as a
    # relative adjustment to the query's own baseline, never import the variant's
    # absolute force as a cross-object anchor.
    force_deltas: dict[str, float] = Field(default_factory=dict)


class RetrievedObjectExperience(BaseModel):
    """One condition selected from a grouped physical-surface neighbor."""

    object_id: str
    surface_id: str | None = None
    condition_id: str = "baseline"
    image_path: str = ""
    mass_g: float | None = None
    roughness_index: float | None = None
    projected_contact_fraction: float | None = None
    semantic_description: str
    gecko_min_force_n: float | None = None
    gecko_feasible: bool | None = None
    silicone_min_force_n: float | None = None
    silicone_feasible: bool | None = None
    score: float
    surface_score: float | None = None
    rank: int = 0
    surface_rank: int = 0
    condition_rank: int = 0
    condition_role: ConditionRole = "baseline"
    comparison_to_baseline: ConditionComparison | None = None
    similarity: SimilarityBreakdown

    def model_post_init(self, __context: object) -> None:
        del __context
        if self.surface_id is None:
            self.surface_id = self.object_id.split("__", 1)[0]
        if self.condition_id == "baseline" and "__" in self.object_id:
            self.condition_id = self.object_id.split("__", 1)[1]
        if self.condition_id != "baseline" and self.condition_role == "baseline":
            self.condition_role = "controlled_variant"

    def to_payload(
        self,
        *,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        active_grippers: tuple[Gripper, ...] = (
            Gripper.GECKO,
            Gripper.SILICONE,
        ),
        include_roughness: bool = True,
        include_contact: bool = True,
    ) -> dict:
        outcomes: dict[str, float | bool | None] = {}
        if Gripper.GECKO in active_grippers:
            outcomes.update(
                gecko_min_force_n=self.gecko_min_force_n,
                gecko_feasible=self.gecko_feasible,
            )
        if Gripper.SILICONE in active_grippers:
            outcomes.update(
                silicone_min_force_n=self.silicone_min_force_n,
                silicone_feasible=self.silicone_feasible,
            )
        if mode is RetrievalMode.SEMANTIC_ONLY:
            return {
                **{key: value for key, value in outcomes.items() if value is not None},
            }
        excluded = {"image_path", "gecko_min_force_n", "gecko_feasible",
                    "silicone_min_force_n", "silicone_feasible"}
        if not include_roughness:
            excluded.add("roughness_index")
        if not include_contact:
            excluded.add("projected_contact_fraction")
        return {
            **self.model_dump(mode="json", exclude=excluded, exclude_none=True),
            **{key: value for key, value in outcomes.items() if value is not None},
        }


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #
class ExperienceIndex:
    """Holds experiences + their (once-computed) embeddings and does exact search."""

    def __init__(self, cfg: Config, provider: EmbeddingProvider | None = None) -> None:
        self.cfg = cfg
        self.provider = provider or get_embedding_provider(cfg)
        self.objects: dict[str, ObjectRecord] = {}
        self.surface_conditions: dict[str, list[ObjectRecord]] = {}
        self.surface_vectors: dict[str, np.ndarray] = {}

    def fit(self, records: list[ExperienceRecord]) -> ExperienceIndex:
        self.objects = group_by_object(records)
        self.surface_conditions = {}
        self.surface_vectors = {}
        for obj in self.objects.values():
            surface_id = obj.surface_id or obj.object_id
            self.surface_conditions.setdefault(surface_id, []).append(obj)
        for conditions in self.surface_conditions.values():
            conditions.sort(key=lambda item: (item.condition_id != "baseline", item.object_id))
        for surface_id, conditions in self.surface_conditions.items():
            obj = conditions[0]
            rec = obj.gecko or obj.silicone
            if rec is None:
                continue
            text = build_embedding_text(rec.semantic_description)
            self.surface_vectors[surface_id] = self.provider.embed(text)
        return self

    def _score(
        self,
        query: Query,
        query_vec: np.ndarray,
        rec: ExperienceRecord,
        reference_vec: np.ndarray,
        mode: RetrievalMode,
        ranking_features: tuple[RankingFeature, ...] | None = None,
    ) -> SimilarityBreakdown:
        semantic = cosine(query_vec, reference_vec)
        if mode is RetrievalMode.SEMANTIC_ONLY:
            return SimilarityBreakdown(
                mode=mode,
                semantic=semantic,
                semantic_contribution=semantic,
                total=semantic,
            )
        features = ranking_features or default_ranking_features(self.cfg)
        active = set(features)
        w = normalized_weights(self.cfg, features)
        mass = None
        if "mass" in active:
            if query.mass_g is None or rec.mass_g is None:
                raise ValueError("hybrid retrieval requires query and reference mass")
            mass = s_mass(query.mass_g, rec.mass_g, self.cfg.retrieval.sigma_mass)
        roughness = None
        if "roughness" in active:
            if query.roughness_index is None or rec.roughness_index is None:
                raise ValueError("hybrid retrieval requires enabled roughness values")
            roughness = s_roughness(
                query.roughness_index, rec.roughness_index, self.cfg
            )
        roughness_contribution = (
            w["roughness"] * roughness if roughness is not None else None
        )
        return SimilarityBreakdown(
            mode=mode,
            semantic=semantic,
            mass=mass,
            roughness=roughness,
            semantic_contribution=w["semantic"] * semantic,
            mass_contribution=w["mass"] * mass if mass is not None else None,
            roughness_contribution=roughness_contribution,
            total=(
                w["semantic"] * semantic
                + (w["mass"] * mass if mass is not None else 0.0)
                + (roughness_contribution or 0.0)
            ),
        )

    def embed_query(self, query: Query) -> np.ndarray:
        text = build_embedding_text(query.semantic_description)
        return self.provider.embed(text, is_query=True)

    def retrieve_objects(
        self,
        query: Query,
        query_vec: np.ndarray,
        k: int | None = None,
        exclude_object_id: str | None = None,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        ranking_features: tuple[RankingFeature, ...] | None = None,
        visible_condition_fields: tuple[MeasurementField, ...] | None = None,
    ) -> list[RetrievedObjectExperience]:
        """Rank surfaces, then retain only experiment-visible controlled conditions."""
        k = k or self.cfg.retrieval.k
        scored_surfaces: list[tuple[float, str, list[RetrievedObjectExperience]]] = []
        for surface_id, conditions in self.surface_conditions.items():
            baseline = next(
                (item for item in conditions if item.condition_id == "baseline"),
                None,
            )
            comparisons: dict[str, ConditionComparison] = {}
            if mode is RetrievalMode.HYBRID and visible_condition_fields is not None:
                if baseline is None:
                    if len(conditions) > 1:
                        raise ValueError(
                            f"surface {surface_id!r} has multiple conditions but no baseline"
                        )
                    continue
                baseline_rec = baseline.gecko or baseline.silicone
                if baseline_rec is None:
                    continue
                visible = set(visible_condition_fields)
                for obj in conditions:
                    if obj is baseline:
                        continue
                    rec = obj.gecko or obj.silicone
                    if rec is None:
                        continue
                    changed = tuple(
                        field
                        for field in MEASUREMENT_FIELDS
                        if _measurement_changed(
                            getattr(baseline_rec, field), getattr(rec, field)
                        )
                    )
                    if not changed or not set(changed).issubset(visible):
                        continue
                    if any(
                        getattr(baseline_rec, field) is None
                        or getattr(rec, field) is None
                        for field in changed
                    ):
                        continue
                    visible_changed = tuple(
                        field for field in visible_condition_fields if field in changed
                    )
                    force_deltas: dict[str, float] = {}
                    for gripper in (Gripper.GECKO, Gripper.SILICONE):
                        base_outcome = baseline.get(gripper)
                        var_outcome = obj.get(gripper)
                        if (
                            base_outcome is not None
                            and var_outcome is not None
                            and base_outcome.min_force_n is not None
                            and var_outcome.min_force_n is not None
                        ):
                            force_deltas[f"{gripper.value}_min_force_delta_n"] = float(
                                var_outcome.min_force_n - base_outcome.min_force_n
                            )
                    comparisons[obj.object_id] = ConditionComparison(
                        changed_fields=visible_changed,
                        unchanged_fields=tuple(
                            field
                            for field in visible_condition_fields
                            if field not in changed
                        ),
                        deltas={
                            field: float(getattr(rec, field) - getattr(baseline_rec, field))
                            for field in visible_changed
                        },
                        force_deltas=force_deltas,
                    )
            condition_results: list[RetrievedObjectExperience] = []
            reference_vec = self.surface_vectors.get(surface_id)
            if reference_vec is None:
                continue
            for obj in conditions:
                if obj.object_id == exclude_object_id:
                    continue
                if not any(
                    obj.get(gripper) is not None
                    for gripper in self.cfg.prediction.active_grippers
                ):
                    continue
                rec = obj.gecko or obj.silicone
                if rec is None:
                    continue
                breakdown = self._score(
                    query,
                    query_vec,
                    rec,
                    reference_vec,
                    mode,
                    ranking_features,
                )
                condition_results.append(
                    RetrievedObjectExperience(
                        object_id=obj.object_id,
                        surface_id=surface_id,
                        condition_id=obj.condition_id,
                        image_path=rec.image_path if mode is RetrievalMode.HYBRID else "",
                        mass_g=rec.mass_g if mode is RetrievalMode.HYBRID else None,
                        roughness_index=(
                            rec.roughness_index
                            if mode is RetrievalMode.HYBRID
                            and self.cfg.inputs.use_roughness
                            else None
                        ),
                        projected_contact_fraction=(
                            rec.projected_contact_fraction
                            if mode is RetrievalMode.HYBRID
                            and self.cfg.inputs.use_projected_contact
                            else None
                        ),
                        semantic_description=rec.semantic_description,
                        gecko_min_force_n=obj.gecko.min_force_n if obj.gecko else None,
                        gecko_feasible=obj.gecko.feasible if obj.gecko else None,
                        silicone_min_force_n=(
                            obj.silicone.min_force_n if obj.silicone else None
                        ),
                        silicone_feasible=(obj.silicone.feasible if obj.silicone else None),
                        score=breakdown.total,
                        condition_role=(
                            "baseline"
                            if obj.condition_id == "baseline"
                            else "controlled_variant"
                        ),
                        comparison_to_baseline=comparisons.get(obj.object_id),
                        similarity=breakdown,
                    )
                )
            if not condition_results:
                continue
            surface_score = max(item.score for item in condition_results)
            if mode is RetrievalMode.HYBRID and visible_condition_fields is not None:
                condition_results = [
                    item
                    for item in condition_results
                    if item.condition_role == "baseline"
                    or item.object_id in comparisons
                ]
                baseline_results = [
                    item for item in condition_results if item.condition_role == "baseline"
                ]
                if not baseline_results:
                    continue
                variants = sorted(
                    (
                        item
                        for item in condition_results
                        if item.condition_role == "controlled_variant"
                    ),
                    key=lambda item: (-item.score, item.condition_id),
                )
                condition_results = [
                    baseline_results[0],
                    *variants[: max(0, self.cfg.retrieval.conditions_per_surface - 1)],
                ]
            else:
                condition_results.sort(key=lambda item: (-item.score, item.object_id))
                condition_results = condition_results[: self.cfg.retrieval.conditions_per_surface]
            scored_surfaces.append(
                (surface_score, surface_id, condition_results)
            )
        scored_surfaces.sort(key=lambda item: (-item[0], item[1]))
        output: list[RetrievedObjectExperience] = []
        for surface_rank, (surface_score, _surface_id, retrieved_conditions) in enumerate(
            scored_surfaces[:k], start=1
        ):
            for condition_rank, item in enumerate(retrieved_conditions, start=1):
                item.surface_score = surface_score
                item.rank = surface_rank
                item.surface_rank = surface_rank
                item.condition_rank = condition_rank
                output.append(item)
        return output


def _measurement_changed(first: float | None, second: float | None) -> bool:
    if first is None or second is None:
        return first is not second
    return not math.isclose(first, second, rel_tol=0.0, abs_tol=1e-9)
