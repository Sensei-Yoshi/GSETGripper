"""Automatic discovery of datasets stored directly under ``data/``."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ..config import Config
from ..contracts import Gripper
from .models import (
    Dataset,
    DatasetCapabilities,
    DatasetObject,
    DatasetPaths,
    GripperOutcome,
    ImageArtifact,
)
from .storage import attach_checkpoints

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
EXCLUDED_DATASET_DIRS = {"cache"}
EXCLUDED_IMAGE_DIRS = {
    "cache",
    "contact_fraction",
    "descriptors",
    "results",
    "runs",
    "run_images",
    "suites",
}
DERIVED_IMAGE_SUFFIXES = (
    "_mask",
    "_cutout",
    "_contact",
    "_spline_overlay",
)


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "object"


def dataset_paths(cfg: Config, dataset_id: str, image_root: Path | None = None) -> DatasetPaths:
    root = cfg.root / "data" / dataset_id
    return DatasetPaths(
        root=root,
        image_root=image_root or (root / "images" if (root / "images").is_dir() else root),
        descriptors=root / "descriptors",
        preparation_manifest=root / "preparation_manifest.json",
        experiences=root / "validation_experiences.jsonl",
        splits=root / "splits.json",
        runs=root / "runs",
        results=root / "results",
        run_images=root / "run_images",
        suites=root / "suites",
        contact_fraction=root / "contact_fraction",
        cache=cfg.root / "data" / "cache" / dataset_id,
    )


def discover_datasets(cfg: Config, data_root: Path | None = None) -> list[Dataset]:
    root = data_root or (cfg.root / "data")
    if not root.exists():
        return []
    datasets = [
        load_dataset(cfg, path)
        for path in sorted(root.iterdir(), key=lambda item: item.name.casefold())
        if path.is_dir()
        and not path.name.startswith(".")
        and path.name.casefold() not in EXCLUDED_DATASET_DIRS
    ]
    return datasets


def get_dataset(cfg: Config, dataset_id: str = "expforce") -> Dataset:
    path = cfg.root / "data" / dataset_id
    if not path.is_dir():
        raise KeyError(f"dataset folder {path} does not exist")
    return load_dataset(cfg, path)


def load_dataset(cfg: Config, root: Path) -> Dataset:
    if (root / "dataset_2gripper.csv").is_file():
        dataset = _load_expforce(cfg, root)
    else:
        dataset = _load_image_folder(cfg, root)
    return attach_checkpoints(dataset)


def _load_expforce(cfg: Config, root: Path) -> Dataset:
    from ..expforce import load_rows

    dataset_cfg = cfg.model_copy(deep=True)
    dataset_cfg.dataset_id = root.name
    rows = load_rows(dataset_cfg)
    paths = dataset_paths(cfg, root.name, root / "images")
    objects: dict[str, DatasetObject] = {}
    for row in rows:
        image_path = paths.image_root / row.image_name
        relative = str(image_path.relative_to(cfg.root))
        objects[row.object_id] = DatasetObject(
            dataset_id=root.name,
            object_id=row.object_id,
            name=row.object_name,
            image=ImageArtifact(
                path=relative,
                sha256=_file_hash(image_path),
                available=image_path.is_file(),
                remote_url=(
                    "https://raw.githubusercontent.com/expforcesubmission/"
                    f"Exp-Force-Website/main/images/{row.image_name}"
                ),
            ),
            mass_g=row.mass_g,
            roughness_class=row.roughness_class,
            projected_contact_fraction=row.projected_contact_fraction,
            gripper_outcomes={
                Gripper.GECKO: GripperOutcome(
                    gripper=Gripper.GECKO,
                    min_force_n=row.gecko_force_n,
                    feasible=row.gecko_feasible,
                    failed_at_limit_n=None if row.gecko_feasible else dataset_cfg.force.limit_n,
                ),
                Gripper.SILICONE: GripperOutcome(
                    gripper=Gripper.SILICONE,
                    min_force_n=row.silicone_force_n,
                    feasible=row.silicone_feasible,
                    failed_at_limit_n=None if row.silicone_feasible else dataset_cfg.force.limit_n,
                ),
            },
        )
    return Dataset(
        dataset_id=root.name,
        display_name=root.name,
        adapter="expforce_paired_csv",
        paths=paths,
        source_fingerprint=_file_hash(root / "dataset_2gripper.csv") or "",
        objects=objects,
        capabilities=DatasetCapabilities(
            has_images=all(item.image.available for item in objects.values()),
            has_measurements=True,
            has_paired_labels=True,
            can_build_experiences=True,
            can_run_pipeline=True,
            can_benchmark=True,
        ),
    )


def _load_image_folder(cfg: Config, root: Path) -> Dataset:
    paths = dataset_paths(cfg, root.name)
    images = _source_images(root)
    objects: dict[str, DatasetObject] = {}
    for image in images:
        candidate = slug(image.stem)
        object_id = candidate
        suffix = 2
        while object_id in objects:
            object_id = f"{candidate}_{suffix}"
            suffix += 1
        objects[object_id] = DatasetObject(
            dataset_id=root.name,
            object_id=object_id,
            name=image.stem.replace("_", " "),
            image=ImageArtifact(
                path=str(image.relative_to(cfg.root)),
                sha256=_file_hash(image),
                available=True,
            ),
        )
    return Dataset(
        dataset_id=root.name,
        display_name=root.name,
        adapter="image_folder",
        paths=paths,
        source_fingerprint=_inventory_hash(root, images),
        objects=objects,
        capabilities=DatasetCapabilities(has_images=bool(objects)),
    )


def _source_images(root: Path) -> list[Path]:
    output: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative_parts = path.relative_to(root).parts[:-1]
        if any(part.casefold() in EXCLUDED_IMAGE_DIRS for part in relative_parts):
            continue
        if path.stem.lower().endswith(DERIVED_IMAGE_SUFFIXES):
            continue
        output.append(path)
    return output


def _file_hash(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _inventory_hash(root: Path, images: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in images:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
