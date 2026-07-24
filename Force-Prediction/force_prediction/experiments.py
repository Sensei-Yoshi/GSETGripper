"""Canonical experiment catalog and explicit strategy implementations.

This is the one file to read to compare E1, E2, E4, E5, and E6. Each strategy
owns only experiment-specific fitting and prediction; shared VLM contracts,
physics equations, residual learners, and retrieval mechanics stay in their
focused modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .config import (
    EXPERIMENT_DEFINITION_VERSION,
    EXPERIMENT_IDS,
    Config,
    ExperimentConfig,
    ExperimentMethod,
)
from .contracts import (
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
from .physics import PhysicsEstimate, PhysicsModel, calibrate
from .prediction import (
    clamp_force,
    physics_predict,
    predictions_from_joint,
    select,
    vlm_predict_joint,
)
from .retrieval import (
    EmbeddingProvider,
    ExperienceIndex,
    RetrievedObjectExperience,
    build_embedding_text,
    get_embedding_provider,
)

GRIPPERS = (Gripper.GECKO, Gripper.SILICONE)


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    method: ExperimentMethod
    label: str
    summary: str
    force_generation_calls: int


EXPERIMENT_CATALOG: dict[str, ExperimentSpec] = {
    "e1": ExperimentSpec(
        "e1",
        ExperimentMethod.JOINT_VLM,
        "Vision-only zero-shot",
        "One joint image-only VLM estimate and explicit gripper recommendation.",
        1,
    ),
    "e2": ExperimentSpec(
        "e2",
        ExperimentMethod.JOINT_VLM_MEASURED,
        "Measured-input zero-shot",
        "One joint VLM estimate using the image and authoritative measurements.",
        1,
    ),
    "e4": ExperimentSpec(
        "e4",
        ExperimentMethod.PAIRED_RETRIEVAL_VLM,
        "Paired-retrieval VLM",
        "One paired-object retrieval and one joint VLM estimate.",
        1,
    ),
    "e5": ExperimentSpec(
        "e5",
        ExperimentMethod.CALIBRATED_PHYSICS,
        "Calibrated physics",
        "Fold-local bounded calibration of the reduced-order physics equations.",
        0,
    ),
    "e6": ExperimentSpec(
        "e6",
        ExperimentMethod.PHYSICS_SEMANTIC_RESIDUAL,
        "Calibrated physics + semantic residual",
        "The E5 physics estimate plus a fold-local learned residual per gripper.",
        0,
    ),
}

if tuple(EXPERIMENT_CATALOG) != EXPERIMENT_IDS:  # fail at import if the catalog drifts
    raise RuntimeError("experiment catalog IDs do not match the validated config contract")


@dataclass
class QueryInput:
    """Everything an experiment may need about one query object."""

    object_id: str
    mass_g: float
    roughness_class: int
    projected_contact_fraction: float
    image_bgr: np.ndarray | None = None
    image_path: str = ""
    semantic_description: str | None = None


@dataclass
class PipelineRunResult:
    """Selection plus evidence and stable experiment provenance."""

    experiment_id: str
    experiment_method: str
    experiment_definition_version: int
    selection: SelectionResult
    semantic_description: str
    retrieved_objects: list[RetrievedObjectExperience]
    physics_estimates: dict[str, dict[str, Any] | None]
    cache_stats: dict[str, Any]


class ExperimentStrategy(ABC):
    """Common lifecycle for one explicit experiment implementation."""

    def __init__(self, cfg: Config, spec: ExperimentSpec, definition: ExperimentConfig) -> None:
        if definition.method is not spec.method:
            raise ValueError(
                f"{spec.experiment_id} must use {spec.method.value!r}, "
                f"not {definition.method.value!r}"
            )
        self.cfg = cfg
        self.spec = spec
        self.definition = definition

    @abstractmethod
    def fit(self, train_records: list[ExperienceRecord]) -> None:
        """Fit only on the current fold's training records."""

    @abstractmethod
    def predict_detailed(self, query_input: QueryInput) -> PipelineRunResult:
        """Run the selected experiment for one query."""

    def _query(self, query_input: QueryInput, *, needs_description: bool) -> Query:
        description = query_input.semantic_description
        if needs_description and not (description or "").strip():
            description = describe(query_input.image_bgr, self.cfg).description
        return Query(
            object_id=query_input.object_id,
            image_path=query_input.image_path,
            mass_g=query_input.mass_g,
            roughness_class=query_input.roughness_class,
            projected_contact_fraction=query_input.projected_contact_fraction,
            semantic_description=description or "",
        )

    def _instruction(self) -> str:
        prompt = self.definition.prompt
        if prompt is None:
            raise RuntimeError(f"{self.spec.experiment_id} has no configured VLM prompt")
        return self.cfg.prompts.experiments[prompt]

    def _result(
        self,
        *,
        selection: SelectionResult,
        description: str,
        retrieved_objects: list[RetrievedObjectExperience] | None = None,
        physics_estimates: dict[str, dict[str, Any] | None] | None = None,
        used_client: bool,
    ) -> PipelineRunResult:
        cache_stats: dict[str, Any] = {}
        if used_client and not self.cfg.models.dry_run:
            cache_stats = get_client(self.cfg).cache_stats()
        return PipelineRunResult(
            experiment_id=self.spec.experiment_id,
            experiment_method=self.spec.method.value,
            experiment_definition_version=EXPERIMENT_DEFINITION_VERSION,
            selection=selection,
            semantic_description=description,
            retrieved_objects=retrieved_objects or [],
            physics_estimates=physics_estimates or {},
            cache_stats=cache_stats,
        )


class JointVLMStrategy(ExperimentStrategy):
    """Shared E1/E2 implementation; the method determines whether measurements are visible."""

    def fit(self, train_records: list[ExperienceRecord]) -> None:
        del train_records

    def predict_detailed(self, query_input: QueryInput) -> PipelineRunResult:
        query = self._query(query_input, needs_description=False)
        include_measured = self.spec.method is ExperimentMethod.JOINT_VLM_MEASURED
        response = vlm_predict_joint(
            self.cfg,
            query,
            query_input.image_bgr,
            [],
            instruction=self._instruction(),
            include_measured=include_measured,
            include_retrieval=False,
        )
        selection = select(
            predictions_from_joint(response),
            model_recommended_gripper=response.recommended_gripper,
            model_recommendation_summary=response.recommendation_summary,
        )
        return self._result(
            selection=selection,
            description=query.semantic_description,
            used_client=True,
        )


class PairedRetrievalVLMStrategy(ExperimentStrategy):
    """E4: one paired-object retrieval followed by one joint VLM force response."""

    def __init__(self, cfg: Config, spec: ExperimentSpec, definition: ExperimentConfig) -> None:
        super().__init__(cfg, spec, definition)
        self.index: ExperienceIndex | None = None

    def fit(self, train_records: list[ExperienceRecord]) -> None:
        provider = get_embedding_provider(self.cfg)
        self.index = ExperienceIndex(self.cfg, provider).fit(train_records)

    def predict_detailed(self, query_input: QueryInput) -> PipelineRunResult:
        if self.index is None:
            raise RuntimeError("fit must be called before E4 prediction")
        query = self._query(query_input, needs_description=True)
        retrieved = self.index.retrieve_objects(
            query,
            self.index.embed_query(query),
            exclude_object_id=query_input.object_id,
        )
        response = vlm_predict_joint(
            self.cfg,
            query,
            query_input.image_bgr,
            retrieved,
            instruction=self._instruction(),
            include_measured=True,
            include_retrieval=True,
        )
        selection = select(
            predictions_from_joint(response),
            model_recommended_gripper=response.recommended_gripper,
            model_recommendation_summary=response.recommendation_summary,
        )
        return self._result(
            selection=selection,
            description=query.semantic_description,
            retrieved_objects=retrieved,
            used_client=True,
        )


class CalibratedPhysicsStrategy(ExperimentStrategy):
    """E5: calibrated analytical equations with no retrieval, embedding, or VLM stage."""

    def __init__(self, cfg: Config, spec: ExperimentSpec, definition: ExperimentConfig) -> None:
        super().__init__(cfg, spec, definition)
        self.physics: PhysicsModel | None = None

    def fit(self, train_records: list[ExperienceRecord]) -> None:
        self.physics = PhysicsModel(calibrate(train_records, self.cfg), self.cfg)

    def predict_detailed(self, query_input: QueryInput) -> PipelineRunResult:
        if self.physics is None:
            raise RuntimeError("fit must be called before E5 prediction")
        query = self._query(query_input, needs_description=False)
        estimates = {
            gripper: self.physics.min_force(
                gripper,
                query.mass_g,
                query.roughness_class,
                query.projected_contact_fraction,
            )
            for gripper in GRIPPERS
        }
        predictions = {
            gripper: physics_predict(self.cfg, gripper, estimate)
            for gripper, estimate in estimates.items()
        }
        return self._result(
            selection=select(predictions),
            description=query.semantic_description,
            physics_estimates={
                gripper.value: asdict(estimate) for gripper, estimate in estimates.items()
            },
            used_client=False,
        )


class PhysicsSemanticResidualStrategy(ExperimentStrategy):
    """E6: the E5 calibrated solve plus a semantic residual regressor per gripper."""

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
                np.asarray(base_rows),
                embedding_array,
                np.asarray(residuals),
            )

    def predict_detailed(self, query_input: QueryInput) -> PipelineRunResult:
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
            predictions[gripper] = self._predict_residual(query, gripper, estimate, query_vector)

        return self._result(
            selection=select(predictions),
            description=query.semantic_description,
            physics_estimates={
                gripper.value: asdict(estimate) for gripper, estimate in estimates.items()
            },
            used_client=self._uses_embeddings,
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


STRATEGY_TYPES: dict[ExperimentMethod, type[ExperimentStrategy]] = {
    ExperimentMethod.JOINT_VLM: JointVLMStrategy,
    ExperimentMethod.JOINT_VLM_MEASURED: JointVLMStrategy,
    ExperimentMethod.PAIRED_RETRIEVAL_VLM: PairedRetrievalVLMStrategy,
    ExperimentMethod.CALIBRATED_PHYSICS: CalibratedPhysicsStrategy,
    ExperimentMethod.PHYSICS_SEMANTIC_RESIDUAL: PhysicsSemanticResidualStrategy,
}


def create_strategy(cfg: Config, experiment_id: str) -> ExperimentStrategy:
    """Resolve a configured experiment ID to its explicit implementation."""
    normalized = experiment_id.lower()
    definition = cfg.experiment(normalized)
    spec = EXPERIMENT_CATALOG[normalized]
    strategy_type = STRATEGY_TYPES[definition.method]
    return strategy_type(cfg, spec, definition)


def experiment_display_name(experiment_id: str) -> str:
    spec = EXPERIMENT_CATALOG[experiment_id.lower()]
    return f"{spec.experiment_id.upper()} — {spec.label}"
