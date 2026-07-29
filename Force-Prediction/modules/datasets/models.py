"""Dataset-level contracts shared by preparation, pipelines, and the viewer."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from ..config import Config
from ..contracts import Gripper
from ..perception import Description


class ImageArtifact(BaseModel):
    """Lazy reference to one source image."""

    path: str
    sha256: str | None = None
    available: bool = True
    remote_url: str | None = None


class DescriptionArtifact(BaseModel):
    value: Description
    source: str
    model: str | None = None
    signature: str
    image_sha256: str | None = None
    updated_at: str


class EmbeddingArtifact(BaseModel):
    status: str = "pending"
    model: str | None = None
    dim: int | None = None
    descriptor_sha256: str | None = None
    cache_key: str | None = None
    vector_sha256: str | None = None


class RoughnessArtifact(BaseModel):
    """Latest dataset-scoped Marigold result for one object image."""

    metadata_path: str
    source_image_sha256: str
    model: str
    mean: float
    median: float
    std: float
    p25: float | None = None
    p75: float | None = None
    uncertainty_mean: float | None = None
    quality_status: str = "unknown"
    quality_warnings: list[str] = Field(default_factory=list)
    updated_at: str


class ContactFractionArtifact(BaseModel):
    """Schema-v2 contact analysis stored alongside one object."""

    summary_path: str
    schema_version: int
    object_height_mm: float
    object_width_mm: float
    geometric_contact_fraction: float = Field(ge=0, le=1)
    combined_contact_fraction: float = Field(ge=0, le=1)
    grasp_feasible: bool
    antipodal_grasp: bool
    contact_floor_applied: bool


class GripperOutcome(BaseModel):
    gripper: Gripper
    min_force_n: float | None = None
    feasible: bool | None = None
    failed_at_limit_n: float | None = None

    @model_validator(mode="after")
    def _validate_partial_outcome(self) -> GripperOutcome:
        if self.feasible is not True and self.min_force_n is not None:
            raise ValueError("only a feasible outcome may have a minimum force")
        return self

    @property
    def complete(self) -> bool:
        return self.feasible is False or (
            self.feasible is True and self.min_force_n is not None
        )


class DatasetObjectMeasurements(BaseModel):
    """Nullable, incrementally editable object measurements and outcome labels."""

    schema_version: int = 2
    object_id: str
    mass_g: float | None = Field(default=None, gt=0)
    roughness_index: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    # Kept only so schema-v1 files remain inspectable. It is never used as an
    # input and is not converted to the numerical index.
    legacy_roughness_class: int | None = Field(default=None, ge=1, le=5)
    projected_contact_fraction: float | None = Field(default=None, ge=0, le=1)
    gecko_feasible: bool | None = None
    gecko_force_n: float | None = Field(default=None, gt=0)
    silicone_feasible: bool | None = None
    silicone_force_n: float | None = Field(default=None, gt=0)
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @model_validator(mode="after")
    def _validate_outcomes(self) -> DatasetObjectMeasurements:
        for feasible, force, name in (
            (self.gecko_feasible, self.gecko_force_n, "gecko"),
            (self.silicone_feasible, self.silicone_force_n, "silicone"),
        ):
            if feasible is not True and force is not None:
                raise ValueError(f"only a feasible {name} outcome may have a minimum force")
        return self


class DatasetObject(BaseModel):
    """One object assembled from immutable source data and derived artifacts."""

    dataset_id: str
    object_id: str
    name: str
    image: ImageArtifact
    image_2: ImageArtifact | None = None
    mass_g: float | None = Field(default=None, gt=0)
    roughness_index: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    legacy_roughness_class: int | None = Field(default=None, ge=1, le=5)
    projected_contact_fraction: float | None = Field(default=None, ge=0, le=1)
    contact_fraction: ContactFractionArtifact | None = None
    description: DescriptionArtifact | None = None
    embedding: EmbeddingArtifact | None = None
    roughness: RoughnessArtifact | None = None
    gripper_outcomes: dict[Gripper, GripperOutcome] = Field(default_factory=dict)


class DatasetCapabilities(BaseModel):
    has_images: bool = False
    has_second_images: bool = False
    has_descriptions: bool = False
    has_embeddings: bool = False
    has_roughness: bool = False
    has_measurements: bool = False
    legacy_roughness_class_count: int = 0
    has_paired_labels: bool = False
    complete_gecko_labels: int = 0
    complete_silicone_labels: int = 0
    complete_pair_count: int = 0
    can_build_experiences: bool = False
    can_estimate_surface_area: bool = False
    can_run_pipeline: bool = False
    can_benchmark: bool = False


class DatasetPaths(BaseModel):
    root: Path
    objects: Path
    image_root: Path
    descriptors: Path
    preparation_manifest: Path
    experiences: Path
    splits: Path
    runs: Path
    results: Path
    run_images: Path
    suites: Path
    cache: Path

    def object_dir(self, object_id: str) -> Path:
        return self.objects / object_id


class Dataset(BaseModel):
    dataset_id: str
    display_name: str
    adapter: str
    paths: DatasetPaths
    source_fingerprint: str
    objects: dict[str, DatasetObject] = Field(default_factory=dict)
    capabilities: DatasetCapabilities = Field(default_factory=DatasetCapabilities)

    def default_active_grippers(self) -> tuple[Gripper, ...]:
        """Availability-aware UI default; unlabeled datasets retain zero-shot use."""
        counts = self.capabilities
        labeled = tuple(
            gripper
            for gripper, count in (
                (Gripper.GECKO, counts.complete_gecko_labels),
                (Gripper.SILICONE, counts.complete_silicone_labels),
            )
            if count > 0
        )
        return labeled or (Gripper.GECKO, Gripper.SILICONE)

    def selectable_grippers(self) -> tuple[Gripper, ...]:
        """Targets backed by labels, or both targets for an unlabeled dataset."""
        return self.default_active_grippers()

    def runtime_config(self, base: Config) -> Config:
        """Return an isolated config whose data/cache paths belong to this dataset."""
        cfg = base.model_copy(deep=True)
        cfg.dataset_id = self.dataset_id
        cfg.paths.experiences = _config_path(cfg.root, self.paths.experiences)
        cfg.paths.images = _config_path(cfg.root, self.paths.image_root)
        cfg.paths.splits = _config_path(cfg.root, self.paths.splits)
        cfg.paths.cache = _config_path(cfg.root, self.paths.cache)
        return cfg

    @property
    def images(self) -> dict[str, ImageArtifact]:
        return {key: item.image for key, item in self.objects.items()}

    @property
    def second_images(self) -> dict[str, ImageArtifact]:
        return {key: item.image_2 for key, item in self.objects.items() if item.image_2 is not None}

    @property
    def descriptions(self) -> dict[str, DescriptionArtifact]:
        return {
            key: item.description
            for key, item in self.objects.items()
            if item.description is not None
        }

    @property
    def embeddings(self) -> dict[str, EmbeddingArtifact]:
        return {
            key: item.embedding for key, item in self.objects.items() if item.embedding is not None
        }

    def summary(self) -> dict:
        roughness_values = [
            item.roughness_index
            for item in self.objects.values()
            if item.roughness_index is not None
        ]
        favored: Counter[str] = Counter()
        for item in self.objects.values():
            candidates = {
                gripper.value: outcome.min_force_n
                for gripper, outcome in item.gripper_outcomes.items()
                if outcome.feasible and outcome.min_force_n is not None
            }
            if candidates:
                favored[min(candidates, key=lambda key: candidates[key])] += 1
        return {
            "dataset_id": self.dataset_id,
            "objects": len(self.objects),
            "second_images": len(self.second_images),
            "experience_rows": sum(len(item.gripper_outcomes) for item in self.objects.values()),
            "source_sha256": self.source_fingerprint,
            "roughness_index": {
                "count": len(roughness_values),
                "min": min(roughness_values, default=None),
                "max": max(roughness_values, default=None),
            },
            "favored_counts": dict(sorted(favored.items())),
            "default_active_grippers": [
                gripper.value for gripper in self.default_active_grippers()
            ],
            "capabilities": self.capabilities.model_dump(mode="json"),
        }


class PreparationStage(StrEnum):
    INDEX = "index"
    DESCRIPTIONS = "descriptions"
    EMBEDDINGS = "embeddings"
    ROUGHNESS = "roughness"
    SURFACE_AREA = "surface_area"
    EXPERIENCES = "experiences"


class StageStatus(BaseModel):
    status: str = "pending"
    completed: int = 0
    total: int = 0
    failed_object: str | None = None
    error: str | None = None
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class PreparationManifest(BaseModel):
    schema_version: int = 4
    dataset_id: str
    source_fingerprint: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    stages: dict[str, StageStatus] = Field(default_factory=dict)
    missing_images: list[str] = Field(default_factory=list)
    missing_second_images: list[str] = Field(default_factory=list)


class PreparedObjectCheckpoint(BaseModel):
    """Dataset-neutral superset of the historical Exp-Force descriptor checkpoint."""

    schema_version: int = 3
    dataset_id: str = "expforce"
    object_id: str
    object_name: str
    image_name: str
    image_path: str
    image_sha256: str | None = None
    descriptor_source: str
    descriptor_model: str | None = None
    descriptor_signature: str
    descriptor: Description
    embedding_status: str = "pending"
    embedding_model: str | None = None
    embedding_dim: int | None = None
    embedding_descriptor_sha256: str | None = None
    embedding_cache_key: str | None = None
    embedding_sha256: str | None = None
    updated_at: str


def _config_path(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())
