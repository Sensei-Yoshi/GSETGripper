"""Gripper-branched hybrid retrieval with paired-row augmentation.

Pipeline: embed each experience once (semantic description + physical text) →
persist; at query time embed the query, then within a HARD gripper branch rank by
    S = w_s*cos + w_m*S_mass + w_r*S_roughness + w_a*S_contact
and take the top-k. Each retrieved experience is augmented with the SAME object's
force on the OTHER gripper (the paired-row enhancement) so the predictor can see
the gecko<->silicone crossover delta. Exact search (dataset < 1k rows); no vector DB.
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
    Gripper,
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
        "contact": raw.contact,
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


class RetrievedExperience(BaseModel):
    record: ExperienceRecord
    score: float
    rank: int = 0
    similarity: SimilarityBreakdown | None = None
    other_gripper_min_force_n: float | None = None
    other_gripper_feasible: bool | None = None

    def to_payload(self, include_paired: bool) -> dict:
        r = self.record
        payload: dict = {
            "object_id": r.object_id,
            "mass_g": r.mass_g,
            "roughness_class": r.roughness_class,
            "projected_contact_fraction": r.projected_contact_fraction,
            "min_force_n": r.min_force_n,
            "feasible": r.feasible,
            "gripper": r.gripper.value,
            "semantic_description": r.semantic_description,
            "retrieval_rank": self.rank,
            "similarity": self.similarity.model_dump() if self.similarity else None,
        }
        if include_paired:
            payload["other_gripper_min_force_n"] = self.other_gripper_min_force_n
            payload["other_gripper_feasible"] = self.other_gripper_feasible
        return payload


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #
class ExperienceIndex:
    """Holds experiences + their (once-computed) embeddings and does exact search."""

    def __init__(self, cfg: Config, provider: EmbeddingProvider | None = None) -> None:
        self.cfg = cfg
        self.provider = provider or get_embedding_provider(cfg)
        self.records: list[ExperienceRecord] = []
        self.objects: dict[str, ObjectRecord] = {}
        self.vectors: dict[str, np.ndarray] = {}

    @staticmethod
    def _key(rec: ExperienceRecord) -> str:
        return f"{rec.object_id}:{rec.gripper.value}"

    def fit(self, records: list[ExperienceRecord]) -> ExperienceIndex:
        self.records = records
        self.objects = group_by_object(records)
        embedded: dict[str, np.ndarray] = {}
        for rec in records:
            text = build_embedding_text(
                rec.semantic_description, rec.mass_g, rec.roughness_class,
                rec.projected_contact_fraction, self.cfg,
            )
            if text not in embedded:
                embedded[text] = self.provider.embed(text)
            self.vectors[self._key(rec)] = embedded[text]
        return self

    def embed_query(self, query: Query) -> np.ndarray:
        text = build_embedding_text(
            query.semantic_description, query.mass_g, query.roughness_class,
            query.projected_contact_fraction, self.cfg,
        )
        return self.provider.embed(text, is_query=True)

    def retrieve(
        self,
        query: Query,
        query_vec: np.ndarray,
        gripper: Gripper,
        k: int | None = None,
        exclude_object_id: str | None = None,
    ) -> list[RetrievedExperience]:
        w = normalized_weights(self.cfg)
        k = k or self.cfg.retrieval.k
        scored: list[RetrievedExperience] = []
        for rec in self.records:
            if rec.gripper is not gripper:  # HARD gripper-branch filter
                continue
            if exclude_object_id is not None and rec.object_id == exclude_object_id:
                continue
            semantic = cosine(query_vec, self.vectors[self._key(rec)])
            mass = s_mass(query.mass_g, rec.mass_g, self.cfg.retrieval.sigma_mass)
            roughness = s_roughness(query.roughness_class, rec.roughness_class, self.cfg)
            contact = s_contact(
                query.projected_contact_fraction,
                rec.projected_contact_fraction,
                self.cfg.retrieval.sigma_contact,
            )
            breakdown = SimilarityBreakdown(
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
            obj = self.objects.get(rec.object_id)
            other_force = obj.other_gripper_force(rec.gripper) if obj else None
            other = obj.get(rec.gripper.other()) if obj else None
            scored.append(
                RetrievedExperience(
                    record=rec,
                    score=breakdown.total,
                    similarity=breakdown,
                    other_gripper_min_force_n=other_force,
                    other_gripper_feasible=(other.feasible if other else None),
                )
            )
        scored.sort(key=lambda x: x.score, reverse=True)
        top = scored[:k]
        for rank, item in enumerate(top, start=1):
            item.rank = rank
        return top
