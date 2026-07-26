"""Dataset-level contracts shared by preparation, pipelines, and the viewer."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

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


class GripperOutcome(BaseModel):
    gripper: Gripper
    min_force_n: float | None = None
    feasible: bool
    failed_at_limit_n: float | None = None


class DatasetObject(BaseModel):
    """One object assembled from immutable source data and derived artifacts."""

    dataset_id: str
    object_id: str
    name: str
    image: ImageArtifact
    mass_g: float | None = Field(default=None, gt=0)
    roughness_class: int | None = Field(default=None, ge=1, le=5)
    projected_contact_fraction: float | None = Field(default=None, ge=0, le=1)
    description: DescriptionArtifact | None = None
    embedding: EmbeddingArtifact | None = None
    gripper_outcomes: dict[Gripper, GripperOutcome] = Field(default_factory=dict)


class DatasetCapabilities(BaseModel):
    has_images: bool = False
    has_descriptions: bool = False
    has_embeddings: bool = False
    has_measurements: bool = False
    has_paired_labels: bool = False
    can_build_experiences: bool = False
    can_run_pipeline: bool = False
    can_benchmark: bool = False


class DatasetPaths(BaseModel):
    root: Path
    image_root: Path
    descriptors: Path
    preparation_manifest: Path
    experiences: Path
    splits: Path
    runs: Path
    results: Path
    run_images: Path
    suites: Path
    contact_fraction: Path
    cache: Path


class Dataset(BaseModel):
    dataset_id: str
    display_name: str
    adapter: str
    paths: DatasetPaths
    source_fingerprint: str
    objects: dict[str, DatasetObject] = Field(default_factory=dict)
    capabilities: DatasetCapabilities = Field(default_factory=DatasetCapabilities)

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
    def descriptions(self) -> dict[str, DescriptionArtifact]:
        return {
            key: item.description
            for key, item in self.objects.items()
            if item.description is not None
        }

    @property
    def embeddings(self) -> dict[str, EmbeddingArtifact]:
        return {
            key: item.embedding
            for key, item in self.objects.items()
            if item.embedding is not None
        }

    def summary(self) -> dict:
        roughness = Counter(
            item.roughness_class
            for item in self.objects.values()
            if item.roughness_class is not None
        )
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
            "experience_rows": sum(len(item.gripper_outcomes) for item in self.objects.values()),
            "source_sha256": self.source_fingerprint,
            "roughness_counts": dict(sorted(roughness.items())),
            "favored_counts": dict(sorted(favored.items())),
            "capabilities": self.capabilities.model_dump(mode="json"),
        }


class PreparationStage(StrEnum):
    INDEX = "index"
    DESCRIPTIONS = "descriptions"
    EMBEDDINGS = "embeddings"
    EXPERIENCES = "experiences"


class StageStatus(BaseModel):
    status: str = "pending"
    completed: int = 0
    total: int = 0
    failed_object: str | None = None
    error: str | None = None
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class PreparationManifest(BaseModel):
    schema_version: int = 3
    dataset_id: str
    source_fingerprint: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    stages: dict[str, StageStatus] = Field(default_factory=dict)
    missing_images: list[str] = Field(default_factory=list)


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
