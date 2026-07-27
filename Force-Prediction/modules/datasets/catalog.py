"""Automatic discovery of datasets stored directly under ``data/``."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from ..config import Config
from ..contracts import Gripper
from .models import (
    ContactFractionArtifact,
    Dataset,
    DatasetCapabilities,
    DatasetObject,
    DatasetPaths,
    GripperOutcome,
    ImageArtifact,
    RoughnessArtifact,
)
from .storage import attach_checkpoints

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
EXCLUDED_DATASET_DIRS = {"cache"}
EXCLUDED_IMAGE_DIRS = {
    "cache",
    "contact_fraction",
    "descriptors",
    "results",
    "roughness",
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
    objects = root / "objects"
    cache = cfg.root / "data" / "cache" / dataset_id
    return DatasetPaths(
        root=root,
        objects=objects,
        image_root=image_root
        or (
            objects
            if objects.is_dir()
            else (root / "images" if (root / "images").is_dir() else root)
        ),
        descriptors=root / "descriptors",
        preparation_manifest=root / "preparation_manifest.json",
        experiences=cache / "experiences.jsonl",
        splits=root / "splits.json",
        runs=root / "runs",
        results=root / "results",
        run_images=root / "run_images",
        suites=root / "suites",
        cache=cache,
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
    if (root / "dataset.csv").is_file():
        dataset = _load_expforce(cfg, root)
    else:
        dataset = _load_image_folder(cfg, root)
    return attach_checkpoints(dataset)


def _load_expforce(cfg: Config, root: Path) -> Dataset:
    from ..expforce import load_rows

    dataset_cfg = cfg.model_copy(deep=True)
    dataset_cfg.dataset_id = root.name
    rows = load_rows(dataset_cfg)
    paths = dataset_paths(cfg, root.name)
    objects: dict[str, DatasetObject] = {}
    for row in rows:
        image_path = _resolve_row_image(root, row.object_id, row.image_name)
        image_2_path = _resolve_row_second_image(
            root,
            row.object_id,
            row.image_name,
            row.image_name_2,
        )
        relative = str(image_path.relative_to(cfg.root))
        item = DatasetObject(
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
                )
                if root.name == "expforce"
                else None,
            ),
            image_2=(
                ImageArtifact(
                    path=str(image_2_path.relative_to(cfg.root)),
                    sha256=_file_hash(image_2_path),
                    available=True,
                )
                if image_2_path is not None
                else None
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
        _attach_contact_summary(cfg, item, paths.object_dir(row.object_id))
        # For paired CSV datasets the labeled source value remains authoritative. The
        # generated contact artifact is attached for provenance/inspection only.
        item.projected_contact_fraction = row.projected_contact_fraction
        _attach_roughness_summary(cfg, item, paths.object_dir(row.object_id))
        objects[row.object_id] = item
    has_second_images = bool(objects) and all(
        item.image_2 is not None and item.image_2.available for item in objects.values()
    )
    return Dataset(
        dataset_id=root.name,
        display_name=root.name,
        adapter="expforce_paired_csv",
        paths=paths,
        source_fingerprint=_file_hash(root / "dataset.csv") or "",
        objects=objects,
        capabilities=DatasetCapabilities(
            has_images=all(item.image.available for item in objects.values()),
            has_second_images=has_second_images,
            has_roughness=bool(objects)
            and all(item.roughness is not None for item in objects.values()),
            has_measurements=True,
            has_paired_labels=True,
            can_build_experiences=True,
            can_estimate_surface_area=has_second_images,
            can_run_pipeline=True,
            can_benchmark=True,
        ),
    )


def _load_image_folder(cfg: Config, root: Path) -> Dataset:
    paths = dataset_paths(cfg, root.name)
    canonical = _canonical_object_images(paths.objects)
    if canonical:
        images = list(canonical.values())
        second_images = {image: _canonical_second_image(image.parent) for image in images}
    else:
        images, second_images = _partition_flat_image_pairs(_source_images(root))
    objects: dict[str, DatasetObject] = {}
    for image in images:
        candidate = (
            image.parent.name
            if canonical and image.parent.parent == paths.objects
            else slug(image.stem)
        )
        object_id = candidate
        suffix = 2
        while object_id in objects:
            object_id = f"{candidate}_{suffix}"
            suffix += 1
        item = DatasetObject(
            dataset_id=root.name,
            object_id=object_id,
            name=(object_id if canonical else image.stem).replace("_", " "),
            image=ImageArtifact(
                path=str(image.relative_to(cfg.root)),
                sha256=_file_hash(image),
                available=True,
            ),
            image_2=(
                ImageArtifact(
                    path=str(second_images[image].relative_to(cfg.root)),
                    sha256=_file_hash(second_images[image]),
                    available=True,
                )
                if second_images.get(image) is not None
                else None
            ),
        )
        _attach_contact_summary(cfg, item, paths.object_dir(object_id))
        _attach_roughness_summary(cfg, item, paths.object_dir(object_id))
        objects[object_id] = item
    has_second_images = bool(objects) and all(
        item.image_2 is not None and item.image_2.available for item in objects.values()
    )
    return Dataset(
        dataset_id=root.name,
        display_name=root.name,
        adapter="image_folder",
        paths=paths,
        source_fingerprint=_inventory_hash(
            root,
            [
                *images,
                *(image for image in second_images.values() if image is not None),
            ],
        ),
        objects=objects,
        capabilities=DatasetCapabilities(
            has_images=bool(objects),
            has_second_images=has_second_images,
            has_roughness=bool(objects)
            and all(item.roughness is not None for item in objects.values()),
            can_estimate_surface_area=has_second_images,
        ),
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


def _canonical_object_images(objects_root: Path) -> dict[str, Path]:
    if not objects_root.is_dir():
        return {}
    images: dict[str, Path] = {}
    for object_dir in sorted(objects_root.iterdir(), key=lambda path: path.name.casefold()):
        if not object_dir.is_dir() or object_dir.name.startswith("."):
            continue
        candidates = [
            path
            for path in sorted(object_dir.glob("image.*"))
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if candidates:
            images[object_dir.name] = candidates[0]
    return images


def _canonical_second_image(object_dir: Path) -> Path | None:
    candidates: list[Path] = []
    for pattern in ("image_2.*", f"{object_dir.name}_2.*"):
        candidates.extend(
            path
            for path in sorted(object_dir.glob(pattern))
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return candidates[0] if candidates else None


def _partition_flat_image_pairs(
    images: list[Path],
) -> tuple[list[Path], dict[Path, Path | None]]:
    """Treat ``<object>_2`` as a second view only when ``<object>`` exists."""
    by_key: dict[tuple[Path, str], list[Path]] = {}
    for path in images:
        by_key.setdefault((path.parent, path.stem.casefold()), []).append(path)

    secondary_paths: set[Path] = set()
    second_by_primary: dict[Path, Path | None] = {}
    for path in images:
        matches = by_key.get((path.parent, f"{path.stem}_2".casefold()), [])
        second = next(
            (candidate for candidate in matches if candidate.suffix.lower() == path.suffix.lower()),
            matches[0] if matches else None,
        )
        second_by_primary[path] = second
        if second is not None:
            secondary_paths.add(second)
    primaries = [path for path in images if path not in secondary_paths]
    return primaries, {path: second_by_primary.get(path) for path in primaries}


def _resolve_row_image(root: Path, object_id: str, image_name: str) -> Path:
    declared = root / image_name
    if declared.is_file():
        return declared
    object_dir = root / "objects" / object_id
    canonical = [
        path
        for path in sorted(object_dir.glob("image.*"))
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if canonical:
        return canonical[0]
    return root / "images" / Path(image_name).name


def _resolve_row_second_image(
    root: Path,
    object_id: str,
    image_name: str,
    image_name_2: str | None,
) -> Path | None:
    if image_name_2:
        declared = root / image_name_2
        if declared.is_file():
            return declared
        nested = root / "images" / Path(image_name_2).name
        if nested.is_file():
            return nested

    canonical = _canonical_second_image(root / "objects" / object_id)
    if canonical is not None:
        return canonical

    primary = Path(image_name)
    default_name = f"{primary.stem}_2{primary.suffix}"
    for candidate in (root / default_name, root / "images" / default_name):
        if candidate.is_file():
            return candidate
    return None


def _attach_contact_summary(cfg: Config, item: DatasetObject, object_dir: Path) -> None:
    path = object_dir / "contact_fraction" / "summary.json"
    if not path.is_file():
        return
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        results = summary["results"]
        artifact = ContactFractionArtifact(
            summary_path=str(path.relative_to(cfg.root)),
            schema_version=int(summary["schema_version"]),
            object_height_mm=float(results["object_height_mm"]),
            object_width_mm=float(results["object_width_mm"]),
            geometric_contact_fraction=float(results["geometric_contact_fraction"]),
            combined_contact_fraction=float(results["combined_contact_fraction"]),
            grasp_feasible=bool(results["grasp_feasible"]),
            antipodal_grasp=bool(results["antipodal_grasp"]),
            contact_floor_applied=bool(results["contact_floor_applied"]),
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return
    item.contact_fraction = artifact
    item.projected_contact_fraction = artifact.combined_contact_fraction


def _attach_roughness_summary(cfg: Config, item: DatasetObject, object_dir: Path) -> None:
    metadata_paths = sorted(
        (object_dir / "roughness").glob("*/metadata.json"),
        reverse=True,
    )
    for path in metadata_paths:
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            source = metadata["source"]
            model = metadata["model"]
            roughness = metadata["roughness"]
            uncertainty = metadata.get("roughness_uncertainty") or {}
            quality = metadata.get("quality") or {}
            item.roughness = RoughnessArtifact(
                metadata_path=str(path.relative_to(cfg.root)),
                source_image_sha256=str(source["image_sha256"]),
                model=str(model["id"]),
                mean=float(roughness["mean"]),
                median=float(roughness["median"]),
                std=float(roughness["std"]),
                p25=(float(roughness["p25"]) if roughness.get("p25") is not None else None),
                p75=(float(roughness["p75"]) if roughness.get("p75") is not None else None),
                uncertainty_mean=(
                    float(uncertainty["mean"]) if uncertainty.get("mean") is not None else None
                ),
                quality_status=str(quality.get("status", "unknown")),
                quality_warnings=[str(value) for value in quality.get("warnings", [])],
                updated_at=str(metadata["created_at"]),
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        return


def _file_hash(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _inventory_hash(root: Path, images: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in images:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
