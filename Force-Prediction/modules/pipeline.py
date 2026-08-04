"""Thin public facade over the explicit strategies in :mod:`experiments`.

Usage per evaluation fold::

    pipe = Pipeline(cfg, "e6").fit(train_records)
    result = pipe.predict(query_input)

All fold-local fitting and experiment-specific behavior lives in the
``modules/experiments/`` package so the public lifecycle stays uniform.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .contracts import (
    ExperienceRecord,
    Gripper,
    PerGripperPrediction,
    SelectionResult,
)
from .experiments import PipelineRunResult, QueryInput, create_strategy

__all__ = [
    "ForcePrediction",
    "Pipeline",
    "PipelineRunResult",
    "QueryInput",
    "predict_gripper_force",
    "query_input_from_object",
]


@dataclass(frozen=True)
class ForcePrediction:
    """One experiment's force estimate for one explicitly requested gripper."""

    experiment_id: str
    gripper: Gripper
    feasible: bool
    force_n: float | None
    prediction: PerGripperPrediction


class Pipeline:
    def __init__(self, cfg: Config, experiment_id: str) -> None:
        self.cfg = cfg
        self.experiment_id = experiment_id.lower()
        self.strategy = create_strategy(cfg, self.experiment_id)

    def fit(self, train_records: list[ExperienceRecord]) -> Pipeline:
        active = set(self.cfg.prediction.active_grippers)
        self.strategy.fit([record for record in train_records if record.gripper in active])
        return self

    def predict(self, query: QueryInput) -> SelectionResult:
        return self.predict_detailed(query).selection

    def predict_detailed(self, query: QueryInput) -> PipelineRunResult:
        return self.strategy.predict_detailed(query)


def predict_gripper_force(
    cfg: Config,
    experiment: int | str,
    gripper: Gripper | str,
    train_records: list[ExperienceRecord],
    query: QueryInput,
) -> ForcePrediction:
    """Run one experiment for one gripper and return its typed force result.

    Retrieval-backed experiments still require their fold-local training records;
    this convenience wrapper only scopes the existing pipeline to a single output.
    """
    if isinstance(experiment, bool):
        raise ValueError("experiment must be an integer ID or an 'eN' string")
    experiment_id = f"e{experiment}" if isinstance(experiment, int) else experiment.lower()
    selected_gripper = Gripper(gripper)
    scoped = cfg.model_copy(deep=True)
    scoped.prediction.active_grippers = (selected_gripper,)
    selection = Pipeline(scoped, experiment_id).fit(train_records).predict(query)
    prediction = selection.candidate_predictions[selected_gripper.value]
    return ForcePrediction(
        experiment_id=experiment_id,
        gripper=selected_gripper,
        feasible=prediction.feasible,
        force_n=prediction.predicted_normal_force_n if prediction.feasible else None,
        prediction=prediction,
    )


def query_input_from_object(records: list[ExperienceRecord], cfg: Config) -> QueryInput:
    """Build a query from an object's rows and load its RGB image when available."""
    if not records:
        raise ValueError("cannot build a query from an empty record list")
    record = records[0]
    image = None
    path = (cfg.root / record.image_path) if record.image_path else None
    if path and Path(path).exists():
        import cv2

        image = cv2.imread(str(path))
    return QueryInput(
        object_id=record.object_id,
        surface_id=record.surface_id,
        condition_id=record.condition_id,
        mass_g=record.mass_g,
        roughness_index=record.roughness_index,
        projected_contact_fraction=record.projected_contact_fraction,
        image_bgr=image,
        image_path=record.image_path,
        semantic_description=record.semantic_description or None,
    )
