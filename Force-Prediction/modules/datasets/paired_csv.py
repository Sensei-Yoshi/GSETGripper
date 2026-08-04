"""Typed reader and writer for paired-gripper CSV datasets."""

from __future__ import annotations

import csv
import re
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ..config import Config
from ..contracts import ExperienceRecord, Gripper, Meta


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"expected True/False, got {value!r}")
    return normalized == "true"


def _parse_optional_bool(value: str | None) -> bool | None:
    return _parse_bool(value) if value and value.strip() else None


def _parse_optional_float(value: str | None) -> float | None:
    return float(value) if value and value.strip() else None


def _parse_split(value: str | None) -> Literal["train", "test", "surface_validation"]:
    normalized = (value or "train").strip().lower()
    if normalized in {"train", "test", "surface_validation"}:
        return normalized  # type: ignore[return-value]
    raise ValueError(f"expected train/test/surface_validation split, got {value!r}")


class PairedCsvRow(BaseModel):
    object_name: str
    image_name: str
    image_name_2: str | None = None
    condition_id: str = "baseline"
    split: Literal["train", "test", "surface_validation"] = "train"
    mass_g: float | None = Field(default=None, gt=0)
    roughness_index: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    projected_contact_fraction: float | None = Field(default=None, ge=0, le=1)
    silicone_force_n: float | None = Field(default=None, gt=0)
    silicone_feasible: bool | None = None
    gecko_force_n: float | None = Field(default=None, gt=0)
    gecko_feasible: bool | None = None
    favored_gripper: str | None = None

    @property
    def surface_id(self) -> str:
        return slug(self.object_name)

    @property
    def object_id(self) -> str:
        if self.condition_id == "baseline":
            return self.surface_id
        return f"{self.surface_id}__{self.condition_id}"

    @model_validator(mode="after")
    def _validate_labels(self) -> PairedCsvRow:
        self.condition_id = (self.condition_id or "baseline").strip().lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", self.condition_id):
            raise ValueError(
                "condition_id must contain only lowercase letters, numbers, and underscores"
            )
        for force, feasible, name in (
            (self.silicone_force_n, self.silicone_feasible, "silicone"),
            (self.gecko_force_n, self.gecko_feasible, "gecko"),
        ):
            if feasible is not True and force is not None:
                raise ValueError(f"only a feasible {name} row may contain force")
        expected = self.expected_favored()
        if self.favored_gripper != expected:
            raise ValueError(
                f"favored_gripper={self.favored_gripper!r}, expected {expected!r}"
            )
        return self

    def expected_favored(self) -> str | None:
        complete = (
            self.silicone_feasible is False
            or (self.silicone_feasible is True and self.silicone_force_n is not None)
        ) and (
            self.gecko_feasible is False
            or (self.gecko_feasible is True and self.gecko_force_n is not None)
        )
        if not complete:
            return None
        candidates: dict[str, float] = {}
        if self.silicone_feasible and self.silicone_force_n is not None:
            candidates["silicone"] = self.silicone_force_n
        if self.gecko_feasible and self.gecko_force_n is not None:
            candidates["gecko"] = self.gecko_force_n
        if not candidates:
            return "none"
        minimum = min(candidates.values())
        winners = [name for name, force in candidates.items() if force == minimum]
        return winners[0] if len(winners) == 1 else "tie"


def source_path(cfg: Config) -> Path:
    return cfg.root / "data" / cfg.dataset_id / "dataset.csv"


def _object_image_relative(cfg: Config, row: PairedCsvRow) -> Path:
    suffix = Path(row.image_name).suffix.lower() or ".png"
    return Path("data") / cfg.dataset_id / "objects" / row.surface_id / f"image{suffix}"


def load_rows(cfg: Config) -> list[PairedCsvRow]:
    with source_path(cfg).open(newline="", encoding="utf-8") as fh:
        raw_rows = list(csv.DictReader(fh))
    rows = [
        PairedCsvRow(
            object_name=row["Object"],
            image_name=row["Image"],
            image_name_2=next(
                (
                    value.strip()
                    for value in (row.get("Image_2"), row.get("image_2"))
                    if value and value.strip()
                ),
                None,
            ),
            condition_id=(
                (row.get("condition_id") or row.get("Condition_ID") or "").strip()
                or "baseline"
            ),
            split=_parse_split(row.get("split")),
            mass_g=_parse_optional_float(row.get("Mass_g")),
            roughness_index=_parse_optional_float(row.get("roughness_index")),
            projected_contact_fraction=_parse_optional_float(
                row.get("projected_contact_fraction")
            ),
            silicone_force_n=_parse_optional_float(row.get("silicone_force_n")),
            silicone_feasible=_parse_optional_bool(row.get("silicone_feasible")),
            gecko_force_n=_parse_optional_float(row.get("gecko_force_n")),
            gecko_feasible=_parse_optional_bool(row.get("gecko_feasible")),
            favored_gripper=row.get("favored_gripper", "").strip().lower() or None,
        )
        for row in raw_rows
    ]
    ids = [row.object_id for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("object names do not produce unique IDs")
    split_by_surface: dict[str, set[str]] = {}
    for row in rows:
        if row.split != "surface_validation":
            split_by_surface.setdefault(row.surface_id, set()).add(row.split)
    mixed = sorted(surface for surface, splits in split_by_surface.items() if len(splits) > 1)
    if mixed:
        raise ValueError(
            "all train/test conditions of a physical surface must share one split: "
            + ", ".join(mixed)
        )
    return rows


def save_rows(path: Path, rows: list[PairedCsvRow]) -> None:
    """Atomically persist validated paired-object source rows."""
    if len({row.object_id for row in rows}) != len(rows):
        raise ValueError("object names do not produce unique IDs")
    columns = [
        "Object",
        "Image",
        *(["Image_2"] if any(row.image_name_2 for row in rows) else []),
        "split",
        "condition_id",
        "Mass_g",
        "roughness_index",
        "projected_contact_fraction",
        "silicone_force_n",
        "silicone_feasible",
        "gecko_force_n",
        "gecko_feasible",
        "favored_gripper",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=path.parent, delete=False
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            payload = {
                "Object": row.object_name,
                "Image": row.image_name,
                "split": row.split,
                "condition_id": row.condition_id,
                "Mass_g": row.mass_g if row.mass_g is not None else "",
                "roughness_index": (
                    row.roughness_index if row.roughness_index is not None else ""
                ),
                "projected_contact_fraction": (
                    row.projected_contact_fraction
                    if row.projected_contact_fraction is not None
                    else ""
                ),
                "silicone_force_n": (
                    row.silicone_force_n if row.silicone_force_n is not None else ""
                ),
                "silicone_feasible": (
                    row.silicone_feasible if row.silicone_feasible is not None else ""
                ),
                "gecko_force_n": row.gecko_force_n if row.gecko_force_n is not None else "",
                "gecko_feasible": (
                    row.gecko_feasible if row.gecko_feasible is not None else ""
                ),
                "favored_gripper": row.favored_gripper or "",
            }
            if "Image_2" in columns:
                payload["Image_2"] = row.image_name_2 or ""
            writer.writerow(payload)
        temporary = Path(fh.name)
    temporary.replace(path)


def to_experiences(
    cfg: Config,
    rows: list[PairedCsvRow],
    descriptions: dict[str, str] | None = None,
) -> list[ExperienceRecord]:
    descriptions = descriptions or {}
    records: list[ExperienceRecord] = []
    for row in rows:
        description = descriptions.get(
            row.surface_id, descriptions.get(row.object_id, row.object_name)
        )
        for gripper, force, feasible in (
            (Gripper.SILICONE, row.silicone_force_n, row.silicone_feasible),
            (Gripper.GECKO, row.gecko_force_n, row.gecko_feasible),
        ):
            if feasible is None or (feasible and force is None):
                continue
            records.append(
                ExperienceRecord(
                    object_id=row.object_id,
                    surface_id=row.surface_id,
                    condition_id=row.condition_id,
                    image_path=str(_object_image_relative(cfg, row)),
                    mass_g=row.mass_g,
                    roughness_index=row.roughness_index,
                    projected_contact_fraction=row.projected_contact_fraction,
                    gripper=gripper,
                    min_force_n=force if feasible else None,
                    feasible=feasible,
                    failed_at_limit_n=None if feasible else cfg.force.limit_n,
                    semantic_description=description,
                    meta=Meta(
                        pad_id=f"{cfg.dataset_id}-dataset",
                        contact_fraction_source="dataset_csv",
                    ),
                )
            )
    return records
