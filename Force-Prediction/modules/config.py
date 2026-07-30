"""Typed loader and validator for config.yaml and prompts.yaml.

`config.yaml` is the source of numerical tuning and experiment assignments;
`prompts.yaml` is the editable source of prompt and gripper-embodiment context.
This module parses both into validated Pydantic models so the rest of the
codebase gets attribute access and fail-fast validation instead of raw dicts.

Usage:
    from modules.config import load_config
    cfg = load_config()                 # finds config.yaml at the repo root
    cfg.retrieval.weights.semantic      # -> 0.40
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from .contracts import Gripper

# Repo root = parent of the modules package directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
DEFAULT_PROMPTS_PATH = REPO_ROOT / "prompts.yaml"


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


class InputsConfig(BaseModel):
    use_roughness: bool = True
    use_projected_contact: bool = True
    roughness_representation: Literal["continuous", "binary"] = "continuous"


class PredictionConfig(BaseModel):
    """Candidate grippers requested from every experiment in this runtime."""

    active_grippers: tuple[Gripper, ...] = (Gripper.GECKO, Gripper.SILICONE)

    @model_validator(mode="after")
    def _validate_active_grippers(self) -> PredictionConfig:
        if not self.active_grippers:
            raise ValueError("prediction.active_grippers must contain at least one gripper")
        if len(self.active_grippers) != len(set(self.active_grippers)):
            raise ValueError("prediction.active_grippers must not contain duplicates")
        if len(self.active_grippers) > len(Gripper):
            raise ValueError("prediction.active_grippers contains too many grippers")
        expected = tuple(gripper for gripper in Gripper if gripper in self.active_grippers)
        if self.active_grippers != expected:
            raise ValueError(
                "prediction.active_grippers must use stable order: gecko, then silicone"
            )
        return self


class GeometryConfig(BaseModel):
    px_per_mm: float = Field(gt=0)
    pad_length_mm: float = Field(gt=0)
    minimum_bend_radius_mm: float = Field(gt=0)
    side_angle_deg: float = Field(gt=0, lt=90)
    minimum_contact_fraction: float = Field(ge=0, le=1)


class RoughnessConfig(BaseModel):
    """Definition of the recorded continuous roughness measurement."""

    metric_name: str = Field(min_length=1)
    units: str = Field(min_length=1)
    higher_is_rougher: bool = True
    characteristic_scale: float = Field(gt=0, allow_inf_nan=False)
    binary_threshold: float = Field(default=1340.0, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_direction(self) -> RoughnessConfig:
        if not self.higher_is_rougher:
            raise ValueError(
                "roughness.higher_is_rougher must be true for the recorded index"
            )
        return self


class RetrievalWeights(BaseModel):
    semantic: float = Field(ge=0)
    mass: float = Field(ge=0)
    roughness: float = Field(ge=0)
    contact: float = Field(ge=0)


class EmbeddingConfig(BaseModel):
    model: str
    dim: int = Field(gt=0)


class RetrievalConfig(BaseModel):
    k: int = Field(gt=0)
    conditions_per_surface: int = Field(default=3, gt=0, le=3)
    weights: RetrievalWeights
    sigma_mass: float = Field(gt=0)
    sigma_contact: float = Field(gt=0)
    embedding: EmbeddingConfig


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


class EvaluationConfig(BaseModel):
    n_folds: int = Field(ge=2)
    within_thresholds_n: list[float]


class Prompts(BaseModel):
    descriptor_system: str
    prediction_system: str
    descriptor: str
    experiments: dict[str, str]
    target_instructions: dict[str, str]

    @model_validator(mode="after")
    def _validate_target_instructions(self) -> Prompts:
        if set(self.target_instructions) != {"single", "joint"}:
            raise ValueError("target_instructions keys must be exactly ['single', 'joint']")
        return self


class EmbodimentContext(BaseModel):
    """Fixed written context for one gripper embodiment."""

    description: str = Field(min_length=1)


class PromptBundle(BaseModel):
    """Editable prompt document stored independently from numerical tuning."""

    prompts: Prompts
    embodiments: dict[str, EmbodimentContext]

    @model_validator(mode="after")
    def _validate_embodiments(self) -> PromptBundle:
        if set(self.embodiments) != {"gecko", "silicone"}:
            raise ValueError("embodiments keys must be exactly ['gecko', 'silicone']")
        return self


class ExperimentMethod(StrEnum):
    """Supported estimators; each maps to one explicit strategy implementation."""

    VISION_VLM = "vision_vlm"
    MEASURED_VLM = "measured_vlm"
    SEMANTIC_RETRIEVAL_VLM = "semantic_retrieval_vlm"
    HYBRID_RETRIEVAL_VLM = "hybrid_retrieval_vlm"


EXPERIMENT_IDS = ("e1", "e2", "e3", "e4", "e5", "e6")
EXPERIMENT_DEFINITION_VERSION = 11


class ExperimentConfig(BaseModel):
    method: ExperimentMethod
    prompt: str | None = None


class Config(BaseModel):
    seed: int
    paths: Paths
    force: ForceConfig
    collection: CollectionConfig
    inputs: InputsConfig
    prediction: PredictionConfig
    geometry: GeometryConfig
    roughness: RoughnessConfig
    retrieval: RetrievalConfig
    physics: PhysicsConfig
    models: ModelsConfig
    evaluation: EvaluationConfig
    prompts: Prompts
    embodiments: dict[str, EmbodimentContext]
    prompts_file: str = "prompts.yaml"
    experiments: dict[str, ExperimentConfig]
    # Runtime dataset selection. It is intentionally not a config.yaml tunable.
    dataset_id: str = "expforce"

    # Resolved at load time so callers get absolute paths regardless of cwd.
    root: Path = REPO_ROOT

    @model_validator(mode="after")
    def _validate_experiments(self) -> Config:
        if set(self.experiments) != set(EXPERIMENT_IDS):
            raise ValueError(f"experiments keys must be exactly {list(EXPERIMENT_IDS)}")

        vlm_methods = {
            ExperimentMethod.VISION_VLM,
            ExperimentMethod.MEASURED_VLM,
            ExperimentMethod.SEMANTIC_RETRIEVAL_VLM,
            ExperimentMethod.HYBRID_RETRIEVAL_VLM,
        }
        for name, experiment in self.experiments.items():
            if experiment.method in vlm_methods:
                if experiment.prompt is None:
                    raise ValueError(f"VLM experiment {name!r} requires a prompt key")
                if experiment.prompt not in self.prompts.experiments:
                    raise ValueError(
                        f"experiment {name!r} references missing prompt {experiment.prompt!r}"
                    )
            elif experiment.prompt is not None:
                raise ValueError(f"non-VLM experiment {name!r} must not configure a prompt")
        unused_prompts = set(self.prompts.experiments) - {
            experiment.prompt for experiment in self.experiments.values() if experiment.prompt
        }
        if unused_prompts:
            raise ValueError(f"unused experiment prompts: {sorted(unused_prompts)}")
        if set(self.embodiments) != {"gecko", "silicone"}:
            raise ValueError("embodiments keys must be exactly ['gecko', 'silicone']")
        return self

    def path(self, key: Literal["experiences", "images", "splits", "cache"]) -> Path:
        """Absolute path for a configured data location."""
        return (self.root / getattr(self.paths, key)).resolve()

    def experiment(self, name: str) -> ExperimentConfig:
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
    if "dry_run" in raw.get("models", {}):
        raise ValueError(
            "models.dry_run is no longer supported; Gemini is the production backend "
            "for descriptions, embeddings, and E1-E6 prediction"
        )
    if "provider" in raw.get("retrieval", {}).get("embedding", {}):
        raise ValueError(
            "retrieval.embedding.provider is no longer supported; Gemini embeddings "
            "are always used"
        )
    prompts_name = str(raw.get("prompts_file", "prompts.yaml"))
    bundle = load_prompt_bundle(path.parent / prompts_name)
    raw["prompts"] = bundle.prompts.model_dump(mode="python")
    raw["embodiments"] = bundle.embodiments
    cfg = Config.model_validate(raw)
    cfg.root = path.parent
    return cfg


def load_prompt_bundle(path: str | Path = DEFAULT_PROMPTS_PATH) -> PromptBundle:
    """Load and validate the editable prompt/embodiment document."""
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as fh:
        return PromptBundle.model_validate(yaml.safe_load(fh))


def save_prompt_bundle(bundle: PromptBundle, path: str | Path = DEFAULT_PROMPTS_PATH) -> None:
    """Validate and atomically persist prompt edits without touching config.yaml."""
    validated = PromptBundle.model_validate(bundle)
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(
        validated.model_dump(mode="python"), sort_keys=False, allow_unicode=True
    )
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as fh:
        fh.write(payload)
        temporary = Path(fh.name)
    temporary.replace(destination)
    load_config.cache_clear()


def prompt_bundle_path(cfg: Config) -> Path:
    return (cfg.root / cfg.prompts_file).resolve()


def prompt_bundle_sha256(cfg: Config) -> str:
    path = prompt_bundle_path(cfg)
    if path.is_file():
        payload = path.read_bytes()
    else:
        payload = json.dumps(
            {
                "prompts": cfg.prompts.model_dump(mode="json"),
                "embodiments": {
                    name: item.model_dump(mode="json")
                    for name, item in cfg.embodiments.items()
                },
            },
            sort_keys=True,
        ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
