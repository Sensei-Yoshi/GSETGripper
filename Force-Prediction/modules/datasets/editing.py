"""Validated measurement and outcome edits for CSV-backed dataset objects."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from ..config import Config
from ..expforce import ExpForceRow, load_rows, save_rows
from .catalog import get_dataset
from .models import Dataset, PreparationStage, StageStatus
from .storage import (
    build_dataset_experiences,
    load_manifest,
    save_manifest,
)


class DatasetObjectEdit(BaseModel):
    """The editable measurements and labels for one paired CSV object."""

    mass_g: float = Field(gt=0)
    roughness_class: int = Field(ge=1, le=5)
    projected_contact_fraction: float = Field(ge=0, le=1)
    silicone_force_n: float | None = Field(default=None, gt=0)
    silicone_feasible: bool
    gecko_force_n: float | None = Field(default=None, gt=0)
    gecko_feasible: bool

    @model_validator(mode="after")
    def _validate_force_fields(self) -> DatasetObjectEdit:
        for feasible, force, gripper in (
            (self.silicone_feasible, self.silicone_force_n, "silicone"),
            (self.gecko_feasible, self.gecko_force_n, "gecko"),
        ):
            if feasible and force is None:
                raise ValueError(f"{gripper} feasible row requires a minimum force")
            if not feasible and force is not None:
                raise ValueError(f"clear {gripper} force when marking it infeasible")
        return self

    def source_row(self, original: ExpForceRow) -> ExpForceRow:
        payload = {
            "object_name": original.object_name,
            "image_name": original.image_name,
            "image_name_2": original.image_name_2,
            "mass_g": self.mass_g,
            "roughness_class": self.roughness_class,
            "projected_contact_fraction": self.projected_contact_fraction,
            "silicone_force_n": self.silicone_force_n,
            "silicone_feasible": self.silicone_feasible,
            "gecko_force_n": self.gecko_force_n,
            "gecko_feasible": self.gecko_feasible,
        }
        unchecked = ExpForceRow.model_construct(**payload, favored_gripper="none")
        return ExpForceRow(**payload, favored_gripper=unchecked.expected_favored())


def update_csv_dataset_object(
    cfg: Config,
    dataset: Dataset,
    object_id: str,
    edit: DatasetObjectEdit,
) -> Dataset:
    """Save an object edit and refresh every mutable dataset-derived record."""
    if dataset.adapter != "expforce_paired_csv":
        raise ValueError("this dataset has no editable dataset.csv")
    if object_id not in dataset.objects:
        raise KeyError(f"unknown object ID {object_id!r}")
    for gripper, force in (
        ("silicone", edit.silicone_force_n),
        ("gecko", edit.gecko_force_n),
    ):
        if force is not None and force > cfg.force.limit_n:
            raise ValueError(
                f"{gripper} force cannot exceed the {cfg.force.limit_n:g} N hardware limit"
            )

    rows = load_rows(cfg)
    positions = [index for index, row in enumerate(rows) if row.object_id == object_id]
    if len(positions) != 1:
        raise ValueError(f"expected exactly one source row for {object_id!r}")
    original_row = rows[positions[0]]
    updated_row = edit.source_row(original_row)
    updated_rows = list(rows)
    updated_rows[positions[0]] = updated_row

    save_rows(dataset.paths.root / "dataset.csv", updated_rows)

    refreshed = get_dataset(cfg, dataset.dataset_id)
    build_dataset_experiences(refreshed, cfg.force.limit_n)
    _refresh_manifest(refreshed)
    return refreshed


def _refresh_manifest(dataset: Dataset) -> None:
    manifest = load_manifest(dataset)
    manifest.source_fingerprint = dataset.source_fingerprint
    manifest.stages[PreparationStage.EXPERIENCES.value] = StageStatus(
        status="complete",
        completed=len(dataset.objects),
        total=len(dataset.objects),
    )
    save_manifest(dataset, manifest)
