"""Validated, atomic measurement and outcome edits for every dataset adapter."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from ..config import Config
from ..expforce import ExpForceRow, load_rows, save_rows
from .catalog import get_dataset
from .models import Dataset, DatasetObjectMeasurements, PreparationStage, StageStatus
from .storage import (
    build_dataset_experiences,
    load_manifest,
    save_manifest,
    write_json_atomic,
)


class DatasetObjectEdit(BaseModel):
    """Nullable measurements and partial outcome labels for one dataset object."""

    mass_g: float | None = Field(default=None, gt=0)
    roughness_index: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    projected_contact_fraction: float | None = Field(default=None, ge=0, le=1)
    silicone_force_n: float | None = Field(default=None, gt=0)
    silicone_feasible: bool | None = None
    gecko_force_n: float | None = Field(default=None, gt=0)
    gecko_feasible: bool | None = None

    @model_validator(mode="after")
    def _validate_force_fields(self) -> DatasetObjectEdit:
        for feasible, force, gripper in (
            (self.silicone_feasible, self.silicone_force_n, "silicone"),
            (self.gecko_feasible, self.gecko_force_n, "gecko"),
        ):
            if feasible is not True and force is not None:
                raise ValueError(f"clear {gripper} force unless its status is feasible")
        return self

    def source_row(self, original: ExpForceRow) -> ExpForceRow:
        unchecked = ExpForceRow.model_construct(
            object_name=original.object_name,
            image_name=original.image_name,
            image_name_2=original.image_name_2,
            mass_g=self.mass_g,
            roughness_index=self.roughness_index,
            legacy_roughness_class=original.legacy_roughness_class,
            projected_contact_fraction=self.projected_contact_fraction,
            silicone_force_n=self.silicone_force_n,
            silicone_feasible=self.silicone_feasible,
            gecko_force_n=self.gecko_force_n,
            gecko_feasible=self.gecko_feasible,
            favored_gripper=None,
        )
        return ExpForceRow(
            object_name=original.object_name,
            image_name=original.image_name,
            image_name_2=original.image_name_2,
            mass_g=self.mass_g,
            roughness_index=self.roughness_index,
            legacy_roughness_class=original.legacy_roughness_class,
            projected_contact_fraction=self.projected_contact_fraction,
            silicone_force_n=self.silicone_force_n,
            silicone_feasible=self.silicone_feasible,
            gecko_force_n=self.gecko_force_n,
            gecko_feasible=self.gecko_feasible,
            favored_gripper=unchecked.expected_favored(),
        )


def update_dataset_object(
    cfg: Config,
    dataset: Dataset,
    object_id: str,
    edit: DatasetObjectEdit,
) -> Dataset:
    """Save one partial edit and refresh all mutable dataset-derived records."""
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

    if dataset.adapter == "expforce_paired_csv":
        rows = load_rows(cfg)
        positions = [index for index, row in enumerate(rows) if row.object_id == object_id]
        if len(positions) != 1:
            raise ValueError(f"expected exactly one source row for {object_id!r}")
        updated_rows = list(rows)
        updated_rows[positions[0]] = edit.source_row(rows[positions[0]])
        save_rows(dataset.paths.root / "dataset.csv", updated_rows)
    elif dataset.adapter == "image_folder":
        measurements = DatasetObjectMeasurements(
            object_id=object_id,
            **edit.model_dump(mode="python"),
        )
        write_json_atomic(
            dataset.paths.object_dir(object_id) / "measurements.json",
            measurements.model_dump(mode="json"),
        )
    else:
        raise ValueError(f"dataset adapter {dataset.adapter!r} is not editable")

    refreshed = get_dataset(cfg, dataset.dataset_id)
    build_dataset_experiences(refreshed, cfg.force.limit_n)
    _refresh_manifest(refreshed)
    return refreshed


def update_csv_dataset_object(
    cfg: Config,
    dataset: Dataset,
    object_id: str,
    edit: DatasetObjectEdit,
) -> Dataset:
    """Backward-compatible alias for the adapter-neutral update operation."""
    return update_dataset_object(cfg, dataset, object_id, edit)


def _refresh_manifest(dataset: Dataset) -> None:
    manifest = load_manifest(dataset)
    manifest.source_fingerprint = dataset.source_fingerprint
    completed = sum(
        any(outcome.complete for outcome in item.gripper_outcomes.values())
        for item in dataset.objects.values()
    )
    manifest.stages[PreparationStage.EXPERIENCES.value] = StageStatus(
        status="complete",
        completed=completed,
        total=len(dataset.objects),
    )
    save_manifest(dataset, manifest)
