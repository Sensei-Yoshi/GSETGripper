"""Shared contracts and lifecycle helpers for explicit experiment strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import (
    EXPERIMENT_DEFINITION_VERSION,
    Config,
    ExperimentConfig,
    ExperimentMethod,
)
from ..contracts import ExperienceRecord, Query, SelectionResult
from ..models.gemini import get_client
from ..perception import describe
from ..prediction import predictions_from_joint, select, vlm_predict_joint
from ..retrieval import (
    ExperienceIndex,
    RetrievalMode,
    RetrievedObjectExperience,
    get_embedding_provider,
)


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    method: ExperimentMethod
    label: str
    summary: str
    force_generation_calls: int
    uses_measurements: bool = False
    retrieval_mode: RetrievalMode | None = None


@dataclass
class QueryInput:
    """Everything an experiment may need about one query object."""

    object_id: str
    mass_g: float | None = None
    roughness_class: int | None = None
    projected_contact_fraction: float | None = None
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
    retrieval_mode: str | None = None
    effective_inputs: tuple[str, ...] = ()


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

    def _validate_measured_query(self, query_input: QueryInput) -> None:
        missing: list[str] = []
        if query_input.mass_g is None:
            missing.append("mass")
        if self.cfg.inputs.use_roughness and query_input.roughness_class is None:
            missing.append("roughness class")
        if (
            self.cfg.inputs.use_projected_contact
            and query_input.projected_contact_fraction is None
        ):
            missing.append("projected contact fraction")
        if missing:
            raise ValueError(
                f"{self.spec.experiment_id.upper()} requires " + ", ".join(missing)
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
        effective_inputs: tuple[str, ...] = (),
    ) -> PipelineRunResult:
        cache_stats: dict[str, Any] = {}
        if used_client:
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
            retrieval_mode=(
                self.spec.retrieval_mode.value if self.spec.retrieval_mode else None
            ),
            effective_inputs=effective_inputs,
        )


class JointVLMExperiment(ExperimentStrategy):
    """Shared E1/E2 behavior with explicit per-ID subclasses."""

    include_measured = False

    def fit(self, train_records: list[ExperienceRecord]) -> None:
        del train_records

    def predict_detailed(self, query_input: QueryInput) -> PipelineRunResult:
        if self.include_measured:
            self._validate_measured_query(query_input)
        query = self._query(query_input, needs_description=False)
        response = vlm_predict_joint(
            self.cfg,
            query,
            query_input.image_bgr,
            [],
            instruction=self._instruction(),
            include_measured=self.include_measured,
            include_retrieval=False,
        )
        selection = select(
            predictions_from_joint(response),
            model_recommended_gripper=response.recommended_gripper,
            model_comparison_evidence=response.comparison_evidence,
            model_recommendation_summary=response.recommendation_summary,
        )
        inputs: tuple[str, ...] = ("object_image", "gripper_context")
        if self.include_measured:
            inputs += ("mass",)
            if self.cfg.inputs.use_roughness:
                inputs += ("roughness",)
            if self.cfg.inputs.use_projected_contact:
                inputs += ("projected_contact",)
        return self._result(
            selection=selection,
            description=query.semantic_description,
            used_client=True,
            effective_inputs=inputs,
        )


class RetrievalVLMExperiment(ExperimentStrategy):
    """Shared E3/E4 paired-object retrieval lifecycle."""

    include_measured = False
    retrieval_mode = RetrievalMode.SEMANTIC_ONLY

    def __init__(self, cfg: Config, spec: ExperimentSpec, definition: ExperimentConfig) -> None:
        super().__init__(cfg, spec, definition)
        self.index: ExperienceIndex | None = None

    def fit(self, train_records: list[ExperienceRecord]) -> None:
        provider = get_embedding_provider(self.cfg)
        self.index = ExperienceIndex(self.cfg, provider).fit(train_records)

    def predict_detailed(self, query_input: QueryInput) -> PipelineRunResult:
        if self.index is None:
            raise RuntimeError(f"fit must be called before {self.spec.experiment_id.upper()} prediction")
        if self.include_measured:
            self._validate_measured_query(query_input)
        query = self._query(query_input, needs_description=True)
        retrieved = self.index.retrieve_objects(
            query,
            self.index.embed_query(query),
            exclude_object_id=query_input.object_id,
            mode=self.retrieval_mode,
        )
        response = vlm_predict_joint(
            self.cfg,
            query,
            query_input.image_bgr,
            retrieved,
            instruction=self._instruction(),
            include_measured=self.include_measured,
            include_retrieval=True,
            retrieval_mode=self.retrieval_mode,
        )
        selection = select(
            predictions_from_joint(response),
            model_recommended_gripper=response.recommended_gripper,
            model_comparison_evidence=response.comparison_evidence,
            model_recommendation_summary=response.recommendation_summary,
        )
        inputs: tuple[str, ...] = (
            "object_image",
            "gripper_context",
            "semantic_descriptor",
            "semantic_experiences",
        )
        if self.include_measured:
            inputs += ("mass",)
            if self.cfg.inputs.use_roughness:
                inputs += ("roughness",)
            if self.cfg.inputs.use_projected_contact:
                inputs += ("projected_contact",)
        return self._result(
            selection=selection,
            description=query.semantic_description,
            retrieved_objects=retrieved,
            used_client=True,
            effective_inputs=inputs,
        )
