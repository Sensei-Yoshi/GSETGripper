"""Typed loader and validator for config.yaml.

`config.yaml` is the single source of tuning: force conventions, retrieval
weights, physics coefficients/bounds, model IDs, prompts, and the experiment
toggle sets. This module parses it into validated Pydantic models so the rest
of the codebase gets attribute access and fail-fast validation instead of raw
dict lookups.

Usage:
    from force_prediction.config import load_config
    cfg = load_config()                 # finds config.yaml at the repo root
    cfg.retrieval.weights.semantic      # -> 0.40
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

# Repo root = parent of the force_prediction package directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


class Paths(BaseModel):
    experiences: str
    images: str
    splits: str
    cache: str


class ForceConfig(BaseModel):
    limit_n: float = Field(gt=0)
    min_n: float = Field(ge=0)
    hold_seconds: float = Field(ge=0)
    lift_height_mm: float = Field(gt=0)
    gravity: float = Field(gt=0)


class CollectionConfig(BaseModel):
    start_n: float = Field(ge=0)
    coarse_step_n: float = Field(gt=0)
    fine_step_n: float = Field(gt=0)
    repeats: int = Field(gt=0)

    @model_validator(mode="after")
    def _coarse_is_not_finer_than_fine(self) -> CollectionConfig:
        if self.coarse_step_n < self.fine_step_n:
            raise ValueError("collection.coarse_step_n must be >= collection.fine_step_n")
        return self


class GeometryConfig(BaseModel):
    pad_height_mm: float = Field(gt=0)


class RoughnessConfig(BaseModel):
    n_classes: int = Field(ge=2)
    ordinal: bool
    labels: dict[int, str]

    @model_validator(mode="after")
    def _labels_cover_classes(self) -> RoughnessConfig:
        expected = set(range(1, self.n_classes + 1))
        if set(self.labels) != expected:
            raise ValueError(f"roughness.labels keys must be exactly {sorted(expected)}")
        return self


class RetrievalWeights(BaseModel):
    semantic: float = Field(ge=0)
    mass: float = Field(ge=0)
    roughness: float = Field(ge=0)
    contact: float = Field(ge=0)


class EmbeddingConfig(BaseModel):
    provider: Literal["gemini", "mock"]
    model: str
    dim: int = Field(gt=0)


class RetrievalConfig(BaseModel):
    k: int = Field(gt=0)
    weights: RetrievalWeights
    sigma_mass: float = Field(gt=0)
    sigma_contact: float = Field(gt=0)
    embedding: EmbeddingConfig
    use_projected_contact: bool = True


class PhysicsBounds(BaseModel):
    alpha0: tuple[float, float]
    decay: tuple[float, float]
    beta0: tuple[float, float]
    beta_decay: tuple[float, float]
    n50: tuple[float, float]


class PhysicsConfig(BaseModel):
    alpha_sil0: float
    alpha_sil_decay: float
    alpha_geo0: float
    alpha_geo_decay: float
    beta0: float
    beta_decay: float
    n50: float = Field(gt=0)
    bounds: PhysicsBounds


class ModelsConfig(BaseModel):
    vlm: str
    temperature: float = Field(ge=0)
    max_retries: int = Field(ge=0)
    cache: bool
    dry_run: bool


class LearningConfig(BaseModel):
    residual_model: Literal["ridge", "gbt", "gp"]
    embedding_pca_dims: int = Field(ge=0)
    ridge_alpha: float = Field(gt=0)
    gbt_n_estimators: int = Field(gt=0)
    gbt_max_depth: int = Field(gt=0)
    gbt_learning_rate: float = Field(gt=0)


class EvaluationConfig(BaseModel):
    n_folds: int = Field(ge=2)
    within_thresholds_n: list[float]


class Prompts(BaseModel):
    descriptor_system: str
    prediction_system: str
    descriptor: str
    experiments: dict[str, str]


class ExperimentToggles(BaseModel):
    use_measured: bool
    use_retrieval: bool
    use_paired_rows: bool
    use_physics: bool
    use_vlm: bool
    use_residual: bool
    prompt: str | None = None


class Config(BaseModel):
    seed: int
    paths: Paths
    force: ForceConfig
    collection: CollectionConfig
    geometry: GeometryConfig
    roughness: RoughnessConfig
    retrieval: RetrievalConfig
    physics: PhysicsConfig
    models: ModelsConfig
    learning: LearningConfig
    evaluation: EvaluationConfig
    prompts: Prompts
    experiments: dict[str, ExperimentToggles]

    # Resolved at load time so callers get absolute paths regardless of cwd.
    root: Path = REPO_ROOT

    @model_validator(mode="after")
    def _vlm_experiments_have_prompts(self) -> Config:
        for name, toggles in self.experiments.items():
            if toggles.use_vlm:
                if toggles.prompt is None:
                    raise ValueError(f"VLM experiment {name!r} requires a prompt key")
                if toggles.prompt not in self.prompts.experiments:
                    raise ValueError(
                        f"experiment {name!r} references missing prompt {toggles.prompt!r}"
                    )
            elif toggles.prompt is not None:
                raise ValueError(f"non-VLM experiment {name!r} must not configure a prompt")
        return self

    def path(self, key: Literal["experiences", "images", "splits", "cache"]) -> Path:
        """Absolute path for a configured data location."""
        return (self.root / getattr(self.paths, key)).resolve()

    def experiment(self, name: str) -> ExperimentToggles:
        key = name.lower()
        if key not in self.experiments:
            raise KeyError(f"unknown experiment {name!r}; have {sorted(self.experiments)}")
        return self.experiments[key]


@lru_cache(maxsize=8)
def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load, validate, and cache the config at `path`."""
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = Config.model_validate(raw)
    cfg.root = path.parent
    return cfg
