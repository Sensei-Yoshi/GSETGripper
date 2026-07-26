"""E6: E5 calibrated physics plus a fold-local semantic residual."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from ..config import Config, ExperimentConfig
from ..contracts import (
    Compatibility,
    ExperienceRecord,
    Gripper,
    PerGripperPrediction,
    Query,
)
from ..learning import ResidualForceModel, base_features
from ..physics import PhysicsEstimate, PhysicsModel, calibrate
from ..prediction import clamp_force, select
from ..retrieval import EmbeddingProvider, build_embedding_text, get_embedding_provider
from .helper import ExperimentSpec, ExperimentStrategy, QueryInput

GRIPPERS = (Gripper.GECKO, Gripper.SILICONE)


class E6Strategy(ExperimentStrategy):
    def __init__(self, cfg: Config, spec: ExperimentSpec, definition: ExperimentConfig) -> None:
        super().__init__(cfg, spec, definition)
        self.physics: PhysicsModel | None = None
        self.provider: EmbeddingProvider | None = None
        self.residual: dict[Gripper, ResidualForceModel] = {}

    @property
    def _uses_embeddings(self) -> bool:
        return self.cfg.learning.embedding_pca_dims > 0

    def fit(self, train_records: list[ExperienceRecord]) -> None:
        self.physics = PhysicsModel(calibrate(train_records, self.cfg), self.cfg)
        if self._uses_embeddings:
            self.provider = get_embedding_provider(self.cfg)
        self._fit_residuals(train_records)

    def _fit_residuals(self, train_records: list[ExperienceRecord]) -> None:
        assert self.physics is not None
        self.residual.clear()
        for gripper in GRIPPERS:
            base_rows: list[list[float]] = []
            embeddings: list[np.ndarray] = []
            residuals: list[float] = []
            for record in train_records:
                if (
                    record.gripper is not gripper
                    or not record.feasible
                    or record.min_force_n is None
                ):
                    continue
                estimate = self.physics.min_force(
                    gripper,
                    record.mass_g,
                    record.roughness_class,
                    record.projected_contact_fraction,
                )
                if not estimate.feasible or estimate.min_force_n is None:
                    continue
                base_rows.append(
                    base_features(
                        record.mass_g,
                        record.roughness_class,
                        record.projected_contact_fraction,
                        estimate.min_force_n,
                        include_contact=self.cfg.inputs.use_projected_contact,
                    )
                )
                residuals.append(record.min_force_n - estimate.min_force_n)
                if self.provider is not None:
                    embeddings.append(
                        self.provider.embed(build_embedding_text(record.semantic_description))
                    )
            if not base_rows:
                continue
            embedding_array = (
                np.asarray(embeddings)
                if embeddings
                else np.empty((len(base_rows), 0), dtype=float)
            )
            self.residual[gripper] = ResidualForceModel(self.cfg).fit(
                np.asarray(base_rows), embedding_array, np.asarray(residuals)
            )

    def predict_detailed(self, query_input: QueryInput):  # noqa: ANN201
        if self.physics is None:
            raise RuntimeError("fit must be called before E6 prediction")
        query = self._query(query_input, needs_description=self._uses_embeddings)
        query_vector = None
        if self.provider is not None:
            query_vector = self.provider.embed(
                build_embedding_text(query.semantic_description), is_query=True
            )

        predictions: dict[Gripper, PerGripperPrediction] = {}
        estimates: dict[Gripper, PhysicsEstimate] = {}
        for gripper in GRIPPERS:
            estimate = self.physics.min_force(
                gripper,
                query.mass_g,
                query.roughness_class,
                query.projected_contact_fraction,
            )
            estimates[gripper] = estimate
            predictions[gripper] = self._predict_residual(
                query, gripper, estimate, query_vector
            )

        return self._result(
            selection=select(predictions),
            description=query.semantic_description,
            physics_estimates={
                gripper.value: asdict(estimate) for gripper, estimate in estimates.items()
            },
            used_client=self._uses_embeddings,
            effective_inputs=(
                "mass",
                "roughness",
                "projected_contact",
                "semantic_embedding",
            ),
        )

    def _predict_residual(
        self,
        query: Query,
        gripper: Gripper,
        estimate: PhysicsEstimate,
        query_vector: np.ndarray | None,
    ) -> PerGripperPrediction:
        if not estimate.feasible or estimate.min_force_n is None:
            return PerGripperPrediction(
                candidate_gripper=gripper,
                feasible=False,
                predicted_normal_force_n=self.cfg.force.limit_n,
                reasoning_trace="calibrated physics declared the grasp infeasible",
            )
        model = self.residual.get(gripper)
        if model is None:
            return PerGripperPrediction(
                candidate_gripper=gripper,
                feasible=True,
                predicted_normal_force_n=estimate.min_force_n,
                reasoning_trace="calibrated physics fallback; no residual training samples",
            )
        base = np.asarray(
            [
                base_features(
                    query.mass_g,
                    query.roughness_class,
                    query.projected_contact_fraction,
                    estimate.min_force_n,
                    include_contact=self.cfg.inputs.use_projected_contact,
                )
            ]
        )
        embedding = (
            query_vector[None, :]
            if query_vector is not None
            else np.empty((1, 0), dtype=float)
        )
        predicted_residual = float(model.predict_residual(base, embedding)[0])
        return PerGripperPrediction(
            candidate_gripper=gripper,
            compatibility=Compatibility.UNKNOWN,
            feasible=True,
            predicted_normal_force_n=clamp_force(
                estimate.min_force_n + predicted_residual, self.cfg
            ),
            reasoning_trace="calibrated physics + learned semantic residual",
        )
