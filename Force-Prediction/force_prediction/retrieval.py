"""Paired-object retrieval over semantics and measured object properties.

E4 ranks each object once and returns both gripper labels for every neighbor.
Exact search is appropriate for this dataset (<1k objects), so no vector database
is required. The hybrid score is only a neighbor-ranking heuristic: it never
evaluates the E5 holding-force equations and never produces a force prediction.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, Field

from .config import Config
from .contracts import (
    ExperienceRecord,
    ObjectRecord,
    Query,
    group_by_object,
)
from .llm import get_client


# --------------------------------------------------------------------------- #
# Embedding providers
# --------------------------------------------------------------------------- #
@runtime_checkable
class EmbeddingProvider(Protocol):
    def embed(
        self, text: str, image_bgr: np.ndarray | None = None, is_query: bool = False
    ) -> np.ndarray: ...


class MockEmbeddingProvider:
    """Deterministic hash-based vectors so offline retrieval is reproducible.

    Ignores `is_query` (a hash mock has no learned query/doc alignment, so the
    asymmetric template would break same-object query<->doc matching)."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def embed(
        self, text: str, image_bgr: np.ndarray | None = None, is_query: bool = False
    ) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-12)


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
    if cfg.models.dry_run or cfg.retrieval.embedding.provider == "mock":
        return MockEmbeddingProvider(cfg.retrieval.embedding.dim)
    return GeminiEmbeddingProvider(cfg)


# --------------------------------------------------------------------------- #
# Embedding text + similarity terms
# --------------------------------------------------------------------------- #
def build_embedding_text(
    description: str,
    mass_g: float | None = None,
    roughness_class: int | None = None,
    contact: float | None = None,
    cfg: Config | None = None,
) -> str:
    """Return semantic-only text for the vector embedding.

    The optional legacy arguments are accepted so older scripts keep working.
    Measured properties are scored explicitly below; including them in the vector
    as well would make the dashboard's retrieval weights impossible to interpret.
    """
    del mass_g, roughness_class, contact, cfg
    return description.strip()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def s_mass(m_q: float, m_i: float, sigma: float) -> float:
    return math.exp(-abs(math.log(m_q) - math.log(m_i)) / sigma)


def s_roughness(r_q: int, r_i: int, cfg: Config) -> float:
    if cfg.roughness.ordinal:
        return 1.0 - abs(r_q - r_i) / (cfg.roughness.n_classes - 1)
    return 1.0 if r_q == r_i else 0.0


def s_contact(a_q: float, a_i: float, sigma: float) -> float:
    return math.exp(-abs(a_q - a_i) / sigma)


def normalized_weights(cfg: Config) -> dict[str, float]:
    raw = cfg.retrieval.weights
    values = {
        "semantic": raw.semantic,
        "mass": raw.mass,
        "roughness": raw.roughness,
        "contact": raw.contact if cfg.inputs.use_projected_contact else 0.0,
    }
    total = sum(values.values())
    if total <= 0:
        raise ValueError("at least one retrieval weight must be positive")
    return {name: value / total for name, value in values.items()}


# --------------------------------------------------------------------------- #
# Retrieved experience (payload-ready)
# --------------------------------------------------------------------------- #
class SimilarityBreakdown(BaseModel):
    semantic: float = Field(ge=-1.0, le=1.0)
    mass: float = Field(ge=0.0, le=1.0)
    roughness: float = Field(ge=0.0, le=1.0)
    contact: float = Field(ge=0.0, le=1.0)
    semantic_contribution: float
    mass_contribution: float
    roughness_contribution: float
    contact_contribution: float
    total: float


class RetrievedObjectExperience(BaseModel):
    """One object-level neighbor with its paired gecko and silicone labels."""

    object_id: str
    image_path: str
    mass_g: float
    roughness_class: int
    projected_contact_fraction: float
    semantic_description: str
    gecko_min_force_n: float | None = None
    gecko_feasible: bool | None = None
    silicone_min_force_n: float | None = None
    silicone_feasible: bool | None = None
    score: float
    rank: int = 0
    similarity: SimilarityBreakdown

    def to_payload(self, *, include_contact: bool = True) -> dict:
        excluded = {"image_path"}
        if not include_contact:
            excluded.add("projected_contact_fraction")
        return self.model_dump(mode="json", exclude=excluded)


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #
class ExperienceIndex:
    """Holds experiences + their (once-computed) embeddings and does exact search."""

    def __init__(self, cfg: Config, provider: EmbeddingProvider | None = None) -> None:
        self.cfg = cfg
        self.provider = provider or get_embedding_provider(cfg)
        self.objects: dict[str, ObjectRecord] = {}
        self.object_vectors: dict[str, np.ndarray] = {}

    def fit(self, records: list[ExperienceRecord]) -> ExperienceIndex:
        self.objects = group_by_object(records)
        embedded: dict[str, np.ndarray] = {}
        for object_id, obj in self.objects.items():
            rec = obj.gecko or obj.silicone
            if rec is None:
                continue
            text = build_embedding_text(
                rec.semantic_description, rec.mass_g, rec.roughness_class,
                rec.projected_contact_fraction, self.cfg,
            )
            if text not in embedded:
                embedded[text] = self.provider.embed(text)
            self.object_vectors[object_id] = embedded[text]
        return self

    def _score(
        self,
        query: Query,
        query_vec: np.ndarray,
        rec: ExperienceRecord,
        reference_vec: np.ndarray,
    ) -> SimilarityBreakdown:
        w = normalized_weights(self.cfg)
        semantic = cosine(query_vec, reference_vec)
        mass = s_mass(query.mass_g, rec.mass_g, self.cfg.retrieval.sigma_mass)
        roughness = s_roughness(query.roughness_class, rec.roughness_class, self.cfg)
        contact = s_contact(
            query.projected_contact_fraction,
            rec.projected_contact_fraction,
            self.cfg.retrieval.sigma_contact,
        )
        return SimilarityBreakdown(
            semantic=semantic,
            mass=mass,
            roughness=roughness,
            contact=contact,
            semantic_contribution=w["semantic"] * semantic,
            mass_contribution=w["mass"] * mass,
            roughness_contribution=w["roughness"] * roughness,
            contact_contribution=w["contact"] * contact,
            total=(
                w["semantic"] * semantic
                + w["mass"] * mass
                + w["roughness"] * roughness
                + w["contact"] * contact
            ),
        )

    def embed_query(self, query: Query) -> np.ndarray:
        text = build_embedding_text(
            query.semantic_description, query.mass_g, query.roughness_class,
            query.projected_contact_fraction, self.cfg,
        )
        return self.provider.embed(text, is_query=True)

    def retrieve_objects(
        self,
        query: Query,
        query_vec: np.ndarray,
        k: int | None = None,
        exclude_object_id: str | None = None,
    ) -> list[RetrievedObjectExperience]:
        """Rank each object once and expose both gripper outcomes in one result."""
        k = k or self.cfg.retrieval.k
        scored: list[RetrievedObjectExperience] = []
        for object_id, obj in self.objects.items():
            if object_id == exclude_object_id:
                continue
            rec = obj.gecko or obj.silicone
            if rec is None:
                continue
            breakdown = self._score(
                query, query_vec, rec, self.object_vectors[object_id]
            )
            scored.append(
                RetrievedObjectExperience(
                    object_id=object_id,
                    image_path=rec.image_path,
                    mass_g=rec.mass_g,
                    roughness_class=rec.roughness_class,
                    projected_contact_fraction=rec.projected_contact_fraction,
                    semantic_description=rec.semantic_description,
                    gecko_min_force_n=obj.gecko.min_force_n if obj.gecko else None,
                    gecko_feasible=obj.gecko.feasible if obj.gecko else None,
                    silicone_min_force_n=obj.silicone.min_force_n if obj.silicone else None,
                    silicone_feasible=obj.silicone.feasible if obj.silicone else None,
                    score=breakdown.total,
                    similarity=breakdown,
                )
            )
        scored.sort(key=lambda item: (-item.score, item.object_id))
        top = scored[:k]
        for rank, item in enumerate(top, start=1):
            item.rank = rank
        return top
