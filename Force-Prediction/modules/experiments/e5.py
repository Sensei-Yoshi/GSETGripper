"""E5: fold-calibrated reduced-order physics."""

from __future__ import annotations

from dataclasses import asdict

from ..config import Config, ExperimentConfig
from ..contracts import ExperienceRecord, Gripper
from ..physics import PhysicsModel, calibrate
from ..prediction import physics_predict, select
from .helper import ExperimentSpec, ExperimentStrategy, QueryInput

GRIPPERS = (Gripper.GECKO, Gripper.SILICONE)


class E5Strategy(ExperimentStrategy):
    def __init__(self, cfg: Config, spec: ExperimentSpec, definition: ExperimentConfig) -> None:
        super().__init__(cfg, spec, definition)
        self.physics: PhysicsModel | None = None

    def fit(self, train_records: list[ExperienceRecord]) -> None:
        self.physics = PhysicsModel(calibrate(train_records, self.cfg), self.cfg)

    def predict_detailed(self, query_input: QueryInput):  # noqa: ANN201
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
            effective_inputs=("mass", "roughness", "projected_contact"),
        )
