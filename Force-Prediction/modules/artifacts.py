"""Current-schema pipeline artifact serialization and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from .config import (
    EXPERIMENT_DEFINITION_VERSION,
    Config,
    prompt_bundle_sha256,
)
from .contracts import ExperienceRecord, Gripper, SelectionResult, load_experiences
from .datasets import get_dataset
from .datasets.storage import dataset_experience_records
from .experiments import EXPERIMENT_CATALOG, experiment_display_name
from .pipeline import PipelineRunResult
from .retrieval import RetrievedObjectExperience

PIPELINE_RUN_SCHEMA_VERSION = 10


def _dataset_root(cfg: Config) -> Path:
    return cfg.root / "data" / cfg.dataset_id


def source_sha256(cfg: Config) -> str:
    return get_dataset(cfg, cfg.dataset_id).source_fingerprint


def load_experience_pool(cfg: Config) -> list[ExperienceRecord]:
    path = cfg.path("experiences")
    if path.exists():
        return load_experiences(path)
    dataset = get_dataset(cfg, cfg.dataset_id)
    return dataset_experience_records(dataset, cfg.force.limit_n)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
        temporary = Path(fh.name)
    temporary.replace(path)


def _object_retrieval_payload(detailed: PipelineRunResult) -> list[dict]:
    return [item.model_dump(mode="json") for item in detailed.retrieved_objects]


def _surface_retrieval_payload(detailed: PipelineRunResult) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for item in detailed.retrieved_objects:
        grouped.setdefault(item.surface_id or item.object_id, []).append(
            item.model_dump(mode="json")
        )
    return [
        {
            "surface_id": surface_id,
            "conditions": sorted(
                conditions,
                key=lambda item: (item.get("condition_rank", 0), item["object_id"]),
            ),
        }
        for surface_id, conditions in sorted(
            grouped.items(),
            key=lambda pair: min(
                item.get("surface_rank", item.get("rank", 0)) for item in pair[1]
            ),
        )
    ]


def pipeline_result_to_dict(detailed: PipelineRunResult) -> dict:
    return {
        "experiment_id": detailed.experiment_id,
        "experiment_method": detailed.experiment_method,
        "experiment_definition_version": detailed.experiment_definition_version,
        "selection": detailed.selection.model_dump(mode="json"),
        "semantic_description": detailed.semantic_description,
        "object_id": detailed.object_id,
        "surface_id": detailed.surface_id,
        "condition_id": detailed.condition_id,
        "retrieval_payload_version": 3,
        "retrieved_objects": _object_retrieval_payload(detailed),
        "retrieved_surfaces": _surface_retrieval_payload(detailed),
        "physics_estimates": detailed.physics_estimates,
        "cache_stats": detailed.cache_stats,
        "active_grippers": list(detailed.active_grippers),
        "generation_mode": detailed.generation_mode,
        "retrieval_mode": detailed.retrieval_mode,
        "ranking_features": list(detailed.ranking_features),
        "visible_condition_fields": list(detailed.visible_condition_fields),
        "condition_policy": detailed.condition_policy,
        "effective_inputs": list(detailed.effective_inputs),
    }


def pipeline_result_from_dict(payload: dict) -> PipelineRunResult:
    selection = SelectionResult.model_validate(payload["selection"])
    active_grippers = tuple(payload["active_grippers"])
    return PipelineRunResult(
        experiment_id=payload["experiment_id"],
        experiment_method=payload["experiment_method"],
        experiment_definition_version=payload["experiment_definition_version"],
        selection=selection,
        semantic_description=payload["semantic_description"],
        retrieved_objects=[
            RetrievedObjectExperience.model_validate(item)
            for item in payload.get("retrieved_objects", [])
        ],
        physics_estimates=payload.get("physics_estimates", {}),
        cache_stats=payload.get("cache_stats", {}),
        active_grippers=active_grippers,
        generation_mode=payload["generation_mode"],
        retrieval_mode=payload.get("retrieval_mode"),
        ranking_features=tuple(payload.get("ranking_features", ())),
        visible_condition_fields=tuple(payload.get("visible_condition_fields", ())),
        condition_policy=payload.get("condition_policy"),
        effective_inputs=tuple(payload.get("effective_inputs", ())),
        object_id=payload.get("object_id", ""),
        surface_id=payload.get("surface_id"),
        condition_id=payload.get("condition_id", "baseline"),
    )


def prompt_provenance(cfg: Config, prompt_key: str | None) -> dict:
    """Exact prompt and fixed embodiment context used by a saved run."""
    embodiments = {
        name: {"description": context.description}
        for name, context in cfg.embodiments.items()
        if Gripper(name) in cfg.prediction.active_grippers
    }
    prediction = None
    if prompt_key is not None:
        target = "single" if len(cfg.prediction.active_grippers) == 1 else "joint"
        prediction = {
            "system": cfg.prompts.prediction_system,
            "instruction_key": prompt_key,
            "instruction": cfg.prompts.experiments[prompt_key],
            "target_instruction_key": target,
            "target_instruction": cfg.prompts.target_instructions[target],
        }
    return {
        "bundle_file": cfg.prompts_file,
        "bundle_sha256": prompt_bundle_sha256(cfg),
        "experiment_instructions": dict(cfg.prompts.experiments),
        "target_instructions": dict(cfg.prompts.target_instructions),
        "prediction": prediction,
        "descriptor": {
            "system": cfg.prompts.descriptor_system,
            "instruction": cfg.prompts.descriptor,
        },
        "embodiments": embodiments,
    }


def backend_provenance(cfg: Config, experiment: str) -> dict[str, str | None]:
    if experiment not in EXPERIMENT_CATALOG:
        raise KeyError(f"unknown experiment {experiment!r}")
    return {
        "force": (
            "gemini_single_generation"
            if len(cfg.prediction.active_grippers) == 1
            else "gemini_joint_generation"
        ),
        "semantic_embedding": (
            cfg.retrieval.embedding.model
            if EXPERIMENT_CATALOG[experiment].retrieval_mode is not None
            else None
        ),
    }


def artifact_backend_label(payload: dict) -> str:
    backend = payload["backend"]
    force = backend.get("force", "unknown")
    embedding = backend.get("semantic_embedding")
    return f"{force} + {embedding}" if embedding else force


def save_pipeline_run(
    cfg: Config,
    *,
    detailed: PipelineRunResult,
    experiment: str,
    query: dict,
    truth: dict | None,
    counterfactual: bool,
    image_bgr=None,  # noqa: ANN001
    baseline: PipelineRunResult | None = None,
) -> Path:
    if detailed.experiment_id != experiment:
        raise ValueError(
            f"result experiment {detailed.experiment_id!r} does not match artifact {experiment!r}"
        )
    created_at = datetime.now(UTC)
    image_path = None
    image_digest = None
    if image_bgr is not None:
        import cv2

        ok, encoded = cv2.imencode(".png", image_bgr)
        if not ok:
            raise RuntimeError("failed to encode run image")
        image_bytes = encoded.tobytes()
        image_digest = hashlib.sha256(image_bytes).hexdigest()
        destination = _dataset_root(cfg) / "run_images" / f"{image_digest}.png"
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as fh:
                fh.write(image_bytes)
                temporary = Path(fh.name)
            temporary.replace(destination)
        image_path = str(destination.relative_to(cfg.root))

    run_id = f"{created_at.strftime('%Y%m%dT%H%M%S%fZ')}_{experiment}_{query['object_id']}"
    definition = cfg.experiment(experiment)
    effective_cfg = EXPERIMENT_CATALOG[experiment].scoped_config(cfg)
    prompt_context = prompt_provenance(cfg, definition.prompt)
    artifact = {
        "schema_version": PIPELINE_RUN_SCHEMA_VERSION,
        "dataset_id": cfg.dataset_id,
        "run_id": run_id,
        "created_at": created_at.isoformat(),
        "source_sha256": source_sha256(cfg),
        "evaluation_protocol": (
            "leave-one-object-out" if truth is not None and not counterfactual else "custom"
        ),
        "experiment": experiment,
        "experiment_method": definition.method.value,
        "experiment_definition_version": EXPERIMENT_DEFINITION_VERSION,
        "experiment_definition": definition.model_dump(mode="json"),
        "prediction_prompts": prompt_context["prediction"],
        "prompt_context": prompt_context,
        "backend": backend_provenance(cfg, experiment),
        "models": {
            "vlm": cfg.models.vlm,
            "embedding": cfg.retrieval.embedding.model,
            "embedding_dim": cfg.retrieval.embedding.dim,
        },
        "inputs": effective_cfg.inputs.model_dump(mode="json"),
        "roughness_measurement": cfg.roughness.model_dump(mode="json"),
        "active_grippers": [
            gripper.value for gripper in cfg.prediction.active_grippers
        ],
        "generation_mode": (
            "single" if len(cfg.prediction.active_grippers) == 1 else "joint"
        ),
        "retrieval_config": cfg.retrieval.model_dump(mode="json"),
        "query": {
            "object_id": detailed.object_id or query.get("object_id", ""),
            "surface_id": detailed.surface_id or query.get("surface_id"),
            "condition_id": detailed.condition_id or query.get("condition_id", "baseline"),
            **query,
            "image_artifact_path": image_path,
            "image_sha256": image_digest,
        },
        "counterfactual": counterfactual,
        "truth": truth,
        "result": pipeline_result_to_dict(detailed),
        "baseline": pipeline_result_to_dict(baseline) if baseline is not None else None,
    }
    path = _dataset_root(cfg) / "runs" / f"{run_id}.json"
    _write_json_atomic(path, artifact)
    return path


def load_saved_runs(cfg: Config) -> list[dict]:
    root = _dataset_root(cfg) / "runs"
    if not root.exists():
        return []
    runs: list[dict] = []
    for path in sorted(root.glob("*.json"), reverse=True):
        try:
            run = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if run.get("schema_version") != PIPELINE_RUN_SCHEMA_VERSION:
            continue
        run["artifact_path"] = str(path)
        run["experiment_display_name"] = saved_run_experiment_label(run)
        run["backend_label"] = artifact_backend_label(run)
        runs.append(run)
    return runs


def saved_run_experiment_label(run: dict) -> str:
    experiment = str(run["experiment"]).lower()
    if experiment not in EXPERIMENT_CATALOG:
        raise KeyError(f"unknown experiment {experiment!r}")
    return experiment_display_name(experiment)
