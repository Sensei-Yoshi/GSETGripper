"""The single, toggle-driven orchestration shared by every experiment.

Experiments E1..E6 differ ONLY by their toggle set (config.yaml); there is one
code path here, which is the anti-duplication keystone of the design.

Usage (per cross-validation fold):
    pipe = Pipeline(cfg, cfg.experiment("e5")).fit(train_records)
    result = pipe.predict(query_input)          # -> SelectionResult

`fit` is where fold-local learning happens (physics calibration, residual model,
retrieval index) so nothing leaks across the GroupKFold boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .config import Config, ExperimentToggles
from .contracts import (
    CandidateQuery,
    Compatibility,
    ExperienceRecord,
    Gripper,
    PerGripperPrediction,
    Query,
    SelectionResult,
)
from .learning import ResidualForceModel, base_features
from .llm import get_client
from .perception import describe
from .physics import PhysicsEstimate, PhysicsModel, PhysicsParams, calibrate
from .prediction import (
    clamp_force,
    physics_predict,
    retrieval_average_predict,
    select,
    vlm_predict_gripper,
    vlm_predict_paired,
)
from .retrieval import (
    EmbeddingProvider,
    ExperienceIndex,
    RetrievedExperience,
    RetrievedObjectExperience,
    build_embedding_text,
    get_embedding_provider,
)

GRIPPERS = (Gripper.GECKO, Gripper.SILICONE)


@dataclass
class QueryInput:
    """Everything the pipeline needs about one query object."""

    object_id: str
    mass_g: float
    roughness_class: int
    projected_contact_fraction: float
    image_bgr: np.ndarray | None = None
    image_path: str = ""
    semantic_description: str | None = None


@dataclass
class PipelineRunResult:
    """Selection plus the evidence needed by evaluation and research tooling."""

    selection: SelectionResult
    semantic_description: str
    retrieved: dict[str, list[RetrievedExperience]]
    retrieved_objects: list[RetrievedObjectExperience]
    physics_estimates: dict[str, dict[str, Any] | None]
    cache_stats: dict[str, Any]


class Pipeline:
    def __init__(self, cfg: Config, toggles: ExperimentToggles) -> None:
        self.cfg = cfg
        self.t = toggles
        self.provider: EmbeddingProvider | None = None
        self.index: ExperienceIndex | None = None
        self.physics: PhysicsModel | None = None
        self.residual: dict[Gripper, ResidualForceModel] = {}

    # ------------------------------------------------------------------ fit #
    def _needs_embeddings(self) -> bool:
        return self.t.use_retrieval or (
            self.t.use_residual and self.cfg.learning.embedding_pca_dims > 0
        )

    def _needs_description(self) -> bool:
        # The force VLM receives the image directly. A separate semantic
        # descriptor is only needed when an embedding-based stage consumes it.
        return self._needs_embeddings()

    def _vlm_instruction(self) -> str:
        if self.t.prompt is None:
            raise RuntimeError("VLM experiment has no configured prompt")
        try:
            return self.cfg.prompts.experiments[self.t.prompt]
        except KeyError as error:
            raise RuntimeError(f"unknown VLM prompt key {self.t.prompt!r}") from error

    def fit(self, train_records: list[ExperienceRecord]) -> Pipeline:
        if self._needs_embeddings():
            self.provider = get_embedding_provider(self.cfg)
        if self.t.use_retrieval:
            self.index = ExperienceIndex(self.cfg, self.provider).fit(train_records)
        if self.t.use_physics or self.t.use_residual:
            params: PhysicsParams = calibrate(train_records, self.cfg)
            self.physics = PhysicsModel(params, self.cfg)
        if self.t.use_residual:
            self._fit_residual(train_records)
        return self

    def _fit_residual(self, train_records: list[ExperienceRecord]) -> None:
        assert self.physics is not None
        for gripper in GRIPPERS:
            base_rows: list[list[float]] = []
            embs: list[np.ndarray] = []
            resid: list[float] = []
            for r in train_records:
                if r.gripper is not gripper or not r.feasible or r.min_force_n is None:
                    continue
                est = self.physics.min_force(
                    gripper, r.mass_g, r.roughness_class, r.projected_contact_fraction
                )
                if not est.feasible or est.min_force_n is None:
                    continue
                base_rows.append(
                    base_features(r.mass_g, r.roughness_class,
                                  r.projected_contact_fraction, est.min_force_n)
                )
                resid.append(r.min_force_n - est.min_force_n)
                if self._needs_embeddings() and self.provider is not None:
                    text = build_embedding_text(
                        r.semantic_description, r.mass_g, r.roughness_class,
                        r.projected_contact_fraction, self.cfg,
                    )
                    embs.append(self.provider.embed(text))
            model = ResidualForceModel(self.cfg)
            if base_rows:
                emb_arr = np.array(embs) if embs else np.empty((len(base_rows), 0))
                model.fit(np.array(base_rows), emb_arr, np.array(resid))
            self.residual[gripper] = model

    # -------------------------------------------------------------- predict #
    def predict(self, q: QueryInput) -> SelectionResult:
        return self.predict_detailed(q).selection

    def predict_detailed(self, q: QueryInput) -> PipelineRunResult:
        description = q.semantic_description
        if description is None and self._needs_description():
            description = describe(q.image_bgr, self.cfg).description
        query = Query(
            object_id=q.object_id,
            image_path=q.image_path,
            mass_g=q.mass_g,
            roughness_class=q.roughness_class,
            projected_contact_fraction=q.projected_contact_fraction,
            semantic_description=description or "",
        )
        query_vec = None
        if self._needs_embeddings() and self.provider is not None:
            query_vec = self.provider.embed(
                build_embedding_text(query.semantic_description, query.mass_g,
                                     query.roughness_class,
                                     query.projected_contact_fraction, self.cfg),
                is_query=True,
            )

        predictions: dict[Gripper, PerGripperPrediction] = {}
        retrieval_trace: dict[str, list[RetrievedExperience]] = {}
        paired_retrieval_trace: list[RetrievedObjectExperience] = []
        physics_trace: dict[str, dict[str, Any] | None] = {}
        if (
            self.t.use_paired_rows
            and self.t.use_retrieval
            and self.t.use_vlm
            and self.index is not None
            and query_vec is not None
        ):
            paired_retrieval_trace = self.index.retrieve_objects(
                query, query_vec, exclude_object_id=q.object_id
            )
            predictions = vlm_predict_paired(
                self.cfg,
                query,
                q.image_bgr,
                paired_retrieval_trace,
                instruction=self._vlm_instruction(),
                include_measured=self.t.use_measured,
            )
            retrieval_trace = {gripper.value: [] for gripper in GRIPPERS}
            physics_trace = {gripper.value: None for gripper in GRIPPERS}
        else:
            for gripper in GRIPPERS:
                cq = CandidateQuery(**query.model_dump(), candidate_gripper=gripper)
                physics_est = None
                if (self.t.use_physics or self.t.use_residual) and self.physics is not None:
                    physics_est = self.physics.min_force(
                        gripper, q.mass_g, q.roughness_class, q.projected_contact_fraction
                    )
                retrieved: list[RetrievedExperience] = []
                if self.t.use_retrieval and self.index is not None and query_vec is not None:
                    retrieved = self.index.retrieve(
                        query, query_vec, gripper, exclude_object_id=q.object_id
                    )
                retrieval_trace[gripper.value] = retrieved
                physics_trace[gripper.value] = (
                    asdict(physics_est) if physics_est is not None else None
                )
                predictions[gripper] = self._predict_one(
                    cq, q.image_bgr, retrieved, physics_est, query_vec
                )
        cache_stats: dict[str, Any] = {}
        if not self.cfg.models.dry_run:
            cache_stats = get_client(self.cfg).cache_stats()
        return PipelineRunResult(
            selection=select(predictions),
            semantic_description=query.semantic_description,
            retrieved=retrieval_trace,
            retrieved_objects=paired_retrieval_trace,
            physics_estimates=physics_trace,
            cache_stats=cache_stats,
        )

    def _predict_one(
        self,
        cq: CandidateQuery,
        image_bgr: np.ndarray | None,
        retrieved: list[RetrievedExperience],
        physics_est: PhysicsEstimate | None,
        query_vec: np.ndarray | None,
    ) -> PerGripperPrediction:
        if self.t.use_residual:
            return self._residual_prediction(cq, physics_est, query_vec)
        if self.t.use_vlm:
            return vlm_predict_gripper(
                self.cfg, cq, image_bgr, retrieved,
                physics_est if self.t.use_physics else None,
                include_paired=self.t.use_paired_rows,
                instruction=self._vlm_instruction(),
                include_retrieval=self.t.use_retrieval,
                include_measured=self.t.use_measured,
            )
        if self.t.use_retrieval:
            return retrieval_average_predict(self.cfg, cq, retrieved)
        if self.t.use_physics and physics_est is not None:
            return physics_predict(self.cfg, cq, physics_est)
        # Should not happen for a well-formed experiment.
        raise RuntimeError("no estimator enabled for this experiment")

    def _residual_prediction(
        self, cq: CandidateQuery, physics_est: PhysicsEstimate | None, query_vec: np.ndarray | None
    ) -> PerGripperPrediction:
        if physics_est is None or not physics_est.feasible or physics_est.min_force_n is None:
            return PerGripperPrediction(
                candidate_gripper=cq.candidate_gripper,
                feasible=False,
                predicted_normal_force_n=self.cfg.force.limit_n,
                reasoning_trace="physics infeasible",
            )
        base = np.array([base_features(cq.mass_g, cq.roughness_class,
                                       cq.projected_contact_fraction, physics_est.min_force_n)])
        if self.cfg.learning.embedding_pca_dims > 0 and query_vec is not None:
            emb = query_vec[None, :]
        else:
            emb = np.empty((1, 0))
        residual = float(self.residual[cq.candidate_gripper].predict_residual(base, emb)[0])
        force = clamp_force(physics_est.min_force_n + residual, self.cfg)
        return PerGripperPrediction(
            candidate_gripper=cq.candidate_gripper,
            compatibility=Compatibility.UNKNOWN,
            feasible=True,
            predicted_normal_force_n=force,
            reasoning_trace="physics prior + learned residual",
        )


def query_input_from_object(records: list[ExperienceRecord], cfg: Config) -> QueryInput:
    """Build a QueryInput from an object's rows (measured props are shared).

    Loads the RGB image from disk if present; otherwise leaves it None (fine for
    mock/dry-run paths, which do not require pixels)."""
    rec = records[0]
    image = None
    from pathlib import Path

    path = (cfg.root / rec.image_path) if rec.image_path else None
    if path and Path(path).exists():
        import cv2

        image = cv2.imread(str(path))
    return QueryInput(
        object_id=rec.object_id,
        mass_g=rec.mass_g,
        roughness_class=rec.roughness_class,
        projected_contact_fraction=rec.projected_contact_fraction,
        image_bgr=image,
        image_path=rec.image_path,
    )
