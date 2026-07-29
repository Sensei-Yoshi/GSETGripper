"""Validated, atomic measurement and outcome edits for every dataset adapter."""

from __future__ import annotations

from typing import Literal

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

    split: Literal["train", "test"] | None = None
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
            condition_id=original.condition_id,
            split=self.split or original.split,
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
            condition_id=original.condition_id,
            split=self.split or original.split,
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
    _validate_edit(cfg, edit)

    if dataset.adapter == "expforce_paired_csv":
        rows = load_rows(cfg)
        positions = [index for index, row in enumerate(rows) if row.object_id == object_id]
        if len(positions) != 1:
            raise ValueError(f"expected exactly one source row for {object_id!r}")
        updated_rows = list(rows)
        updated = edit.source_row(rows[positions[0]])
        measurement_tuple = (
            updated.mass_g,
            updated.roughness_index,
            updated.projected_contact_fraction,
        )
        if any(
            index != positions[0]
            and row.surface_id == updated.surface_id
            and (row.mass_g, row.roughness_index, row.projected_contact_fraction)
            == measurement_tuple
            for index, row in enumerate(rows)
        ):
            raise ValueError(
                "an identical mass/roughness/contact measurement tuple already exists"
            )
        updated_rows = [
            (
                row.model_copy(update={"split": updated.split})
                if row.surface_id == updated.surface_id
                else row
            )
            for row in updated_rows
        ]
        updated_rows[positions[0]] = updated
        save_rows(dataset.paths.root / "dataset.csv", updated_rows)
    elif dataset.adapter == "image_folder":
        measurements = DatasetObjectMeasurements(
            object_id=object_id,
            split=edit.split or dataset.objects[object_id].split,
            **edit.model_dump(mode="python", exclude={"split"}),
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


def add_dataset_condition(
    cfg: Config,
    dataset: Dataset,
    surface_id: str,
    edit: DatasetObjectEdit,
    *,
    condition_id: str | None = None,
) -> tuple[Dataset, str]:
    """Atomically append one independently measured condition to a CSV surface."""
    if dataset.adapter != "expforce_paired_csv":
        raise ValueError("additional conditions currently require a paired CSV dataset")
    _validate_edit(cfg, edit, require_enabled_measurements=True)
    rows = load_rows(cfg)
    surface_rows = [row for row in rows if row.surface_id == surface_id]
    if not surface_rows:
        raise KeyError(f"unknown physical surface ID {surface_id!r}")
    new_condition_id = condition_id or _next_condition_id(surface_rows)
    if any(row.condition_id == new_condition_id for row in surface_rows):
        raise ValueError(
            f"condition ID {new_condition_id!r} already exists for {surface_id!r}"
        )
    measurement_tuple = (
        edit.mass_g,
        edit.roughness_index,
        edit.projected_contact_fraction,
    )
    if any(
        (row.mass_g, row.roughness_index, row.projected_contact_fraction)
        == measurement_tuple
        for row in surface_rows
    ):
        raise ValueError("an identical mass/roughness/contact measurement tuple already exists")

    baseline = next(
        (row for row in surface_rows if row.condition_id == "baseline"),
        surface_rows[0],
    )
    template = baseline.model_copy(update={"condition_id": new_condition_id})
    new_row = edit.source_row(template)
    save_rows(dataset.paths.root / "dataset.csv", [*rows, new_row])
    refreshed = _rebuild(cfg, dataset.dataset_id)
    return refreshed, new_row.object_id


def delete_dataset_condition(
    cfg: Config,
    dataset: Dataset,
    object_id: str,
) -> Dataset:
    """Atomically delete an added condition; physical-surface baselines are immutable."""
    if dataset.adapter != "expforce_paired_csv":
        raise ValueError("condition deletion currently requires a paired CSV dataset")
    item = dataset.objects.get(object_id)
    if item is None:
        raise KeyError(f"unknown object ID {object_id!r}")
    if item.condition_id == "baseline":
        raise ValueError("the baseline condition cannot be deleted")
    rows = load_rows(cfg)
    retained = [row for row in rows if row.object_id != object_id]
    if len(retained) != len(rows) - 1:
        raise ValueError(f"expected exactly one source row for {object_id!r}")
    save_rows(dataset.paths.root / "dataset.csv", retained)
    return _rebuild(cfg, dataset.dataset_id)


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


def _validate_edit(
    cfg: Config,
    edit: DatasetObjectEdit,
    *,
    require_enabled_measurements: bool = False,
) -> None:
    for gripper, force in (
        ("silicone", edit.silicone_force_n),
        ("gecko", edit.gecko_force_n),
    ):
        if force is not None and force > cfg.force.limit_n:
            raise ValueError(
                f"{gripper} force cannot exceed the {cfg.force.limit_n:g} N hardware limit"
            )
    if not require_enabled_measurements:
        return
    if edit.mass_g is None:
        raise ValueError("mass is required for a new measurement condition")
    if cfg.inputs.use_roughness and edit.roughness_index is None:
        raise ValueError("roughness index is required while measured roughness is enabled")
    if cfg.inputs.use_projected_contact and edit.projected_contact_fraction is None:
        raise ValueError(
            "projected contact fraction is required while contact mode is enabled"
        )


def _next_condition_id(rows: list[ExpForceRow]) -> str:
    used = {row.condition_id for row in rows}
    prior_numbers = [
        int(row.condition_id.removeprefix("condition_"))
        for row in rows
        if row.condition_id.startswith("condition_")
        and row.condition_id.removeprefix("condition_").isdigit()
    ]
    number = max(prior_numbers, default=1) + 1
    while f"condition_{number}" in used:
        number += 1
    return f"condition_{number}"


def _rebuild(cfg: Config, dataset_id: str) -> Dataset:
    refreshed = get_dataset(cfg, dataset_id)
    build_dataset_experiences(refreshed, cfg.force.limit_n)
    _refresh_manifest(refreshed)
    return refreshed
