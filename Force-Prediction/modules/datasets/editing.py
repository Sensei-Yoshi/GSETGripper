"""Validated, atomic measurement and outcome edits for every dataset adapter."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field, model_validator

from ..config import Config
from .catalog import get_dataset
from .models import Dataset, DatasetObjectMeasurements, PreparationStage, StageStatus
from .paired_csv import PairedCsvRow, load_rows, save_rows
from .storage import (
    build_dataset_experiences,
    load_manifest,
    save_manifest,
    write_json_atomic,
)


class DatasetObjectEdit(BaseModel):
    """Nullable measurements and partial outcome labels for one dataset object."""

    split: Literal["train", "test", "surface_validation"] | None = None
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

    def source_row(self, original: PairedCsvRow) -> PairedCsvRow:
        unchecked = PairedCsvRow.model_construct(
            object_name=original.object_name,
            image_name=original.image_name,
            image_name_2=original.image_name_2,
            condition_id=original.condition_id,
            split=self.split or original.split,
            mass_g=self.mass_g,
            roughness_index=self.roughness_index,
            projected_contact_fraction=self.projected_contact_fraction,
            silicone_force_n=self.silicone_force_n,
            silicone_feasible=self.silicone_feasible,
            gecko_force_n=self.gecko_force_n,
            gecko_feasible=self.gecko_feasible,
            favored_gripper=None,
        )
        return PairedCsvRow(
            object_name=original.object_name,
            image_name=original.image_name,
            image_name_2=original.image_name_2,
            condition_id=original.condition_id,
            split=self.split or original.split,
            mass_g=self.mass_g,
            roughness_index=self.roughness_index,
            projected_contact_fraction=self.projected_contact_fraction,
            silicone_force_n=self.silicone_force_n,
            silicone_feasible=self.silicone_feasible,
            gecko_force_n=self.gecko_force_n,
            gecko_feasible=self.gecko_feasible,
            favored_gripper=unchecked.expected_favored(),
        )


def add_dataset_object(
    cfg: Config,
    dataset: Dataset,
    object_name: str,
    primary_image: bytes,
    edit: DatasetObjectEdit,
    *,
    primary_filename: str,
    secondary_image: bytes | None = None,
    secondary_filename: str | None = None,
) -> tuple[Dataset, str]:
    """Add a new physical surface and its baseline condition to a dataset."""
    name = object_name.strip()
    if not name:
        raise ValueError("object name is required")
    object_id = _object_slug(name)
    if object_id in dataset.objects:
        raise ValueError(f"object ID {object_id!r} already exists")

    _validate_edit(cfg, edit)
    primary_suffix = _validated_image_suffix(primary_image, primary_filename)
    secondary_suffix = None
    if secondary_image is not None:
        secondary_suffix = _validated_image_suffix(
            secondary_image,
            secondary_filename or "image_2",
        )

    objects_root = dataset.paths.objects
    objects_root.mkdir(parents=True, exist_ok=True)
    target = objects_root / object_id
    if target.exists():
        raise ValueError(f"object storage for {object_id!r} already exists")

    stage = Path(tempfile.mkdtemp(prefix=f".{object_id}-upload-", dir=objects_root))
    try:
        primary_path = stage / f"image{primary_suffix}"
        primary_path.write_bytes(primary_image)
        secondary_path = None
        if secondary_image is not None and secondary_suffix is not None:
            secondary_path = stage / f"image_2{secondary_suffix}"
            secondary_path.write_bytes(secondary_image)

        if dataset.adapter == "image_folder":
            measurements = DatasetObjectMeasurements(
                object_id=object_id,
                split=edit.split or "train",
                **edit.model_dump(mode="python", exclude={"split"}),
            )
            write_json_atomic(
                stage / "measurements.json",
                measurements.model_dump(mode="json"),
            )
            uses_canonical_layout = not dataset.objects or any(
                (cfg.root / item.image.path).parent.parent == objects_root
                for item in dataset.objects.values()
            )
            if uses_canonical_layout:
                stage.replace(target)
            else:
                flat_primary = dataset.paths.root / f"{object_id}{primary_suffix}"
                flat_secondary = (
                    dataset.paths.root / f"{object_id}_2{secondary_suffix}"
                    if secondary_suffix is not None
                    else None
                )
                if flat_primary.exists() or (
                    flat_secondary is not None and flat_secondary.exists()
                ):
                    raise ValueError(f"image storage for {object_id!r} already exists")
                primary_path.replace(flat_primary)
                try:
                    if secondary_path is not None and flat_secondary is not None:
                        secondary_path.replace(flat_secondary)
                    stage.replace(target)
                except Exception:
                    flat_primary.unlink(missing_ok=True)
                    if flat_secondary is not None:
                        flat_secondary.unlink(missing_ok=True)
                    raise
        elif dataset.adapter == "paired_csv":
            rows = load_rows(cfg)
            if any(row.surface_id == object_id for row in rows):
                raise ValueError(f"object ID {object_id!r} already exists")
            template = PairedCsvRow.model_construct(
                object_name=name,
                image_name=f"objects/{object_id}/{primary_path.name}",
                image_name_2=(
                    f"objects/{object_id}/{secondary_path.name}"
                    if secondary_path is not None
                    else None
                ),
                condition_id="baseline",
                split=edit.split or "train",
                mass_g=None,
                roughness_index=None,
                projected_contact_fraction=None,
                silicone_force_n=None,
                silicone_feasible=None,
                gecko_force_n=None,
                gecko_feasible=None,
                favored_gripper=None,
            )
            new_row = edit.source_row(template)
            stage.replace(target)
            try:
                save_rows(dataset.paths.root / "dataset.csv", [*rows, new_row])
            except Exception:
                shutil.rmtree(target, ignore_errors=True)
                raise
        else:
            raise ValueError(f"dataset adapter {dataset.adapter!r} does not accept uploads")
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)

    return _rebuild(cfg, dataset.dataset_id), object_id


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

    if dataset.adapter == "paired_csv":
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
    if dataset.adapter != "paired_csv":
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
    if dataset.adapter != "paired_csv":
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


def _next_condition_id(rows: list[PairedCsvRow]) -> str:
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


def _object_slug(value: str) -> str:
    from .paired_csv import slug

    object_id = slug(value)
    if not object_id:
        raise ValueError("object name must contain at least one letter or number")
    return object_id


def _validated_image_suffix(data: bytes, filename: str) -> str:
    if not data:
        raise ValueError(f"{filename!r} is empty")
    if len(data) > 25 * 1024 * 1024:
        raise ValueError(f"{filename!r} exceeds the 25 MB upload limit")
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.size == 0:
        raise ValueError(f"{filename!r} is not a readable image")
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    raise ValueError(f"{filename!r} must be a PNG, JPEG, or WebP image")


def _rebuild(cfg: Config, dataset_id: str) -> Dataset:
    refreshed = get_dataset(cfg, dataset_id)
    build_dataset_experiences(refreshed, cfg.force.limit_n)
    _refresh_manifest(refreshed)
    return refreshed
