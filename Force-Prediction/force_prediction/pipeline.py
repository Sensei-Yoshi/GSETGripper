"""Thin public facade over the explicit strategies in :mod:`experiments`.

Usage per evaluation fold::

    pipe = Pipeline(cfg, "e4").fit(train_records)
    result = pipe.predict(query_input)

All fold-local fitting and experiment-specific behavior lives in
``force_prediction/experiments.py`` so the public lifecycle stays uniform.
"""

from __future__ import annotations

from pathlib import Path

from .config import Config
from .contracts import ExperienceRecord, SelectionResult
from .experiments import PipelineRunResult, QueryInput, create_strategy

__all__ = ["Pipeline", "PipelineRunResult", "QueryInput", "query_input_from_object"]


class Pipeline:
    def __init__(self, cfg: Config, experiment_id: str) -> None:
        self.cfg = cfg
        self.experiment_id = experiment_id.lower()
        self.strategy = create_strategy(cfg, self.experiment_id)

    def fit(self, train_records: list[ExperienceRecord]) -> Pipeline:
        self.strategy.fit(train_records)
        return self

    def predict(self, query: QueryInput) -> SelectionResult:
        return self.predict_detailed(query).selection

    def predict_detailed(self, query: QueryInput) -> PipelineRunResult:
        return self.strategy.predict_detailed(query)


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
        mass_g=record.mass_g,
        roughness_class=record.roughness_class,
        projected_contact_fraction=record.projected_contact_fraction,
        image_bgr=image,
        image_path=record.image_path,
        semantic_description=record.semantic_description or None,
    )
