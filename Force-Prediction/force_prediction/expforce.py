"""Exp-Force experience-pool preparation, inspection, and benchmark tools."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import tempfile
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from .config import EXPERIMENT_DEFINITION_VERSION, Config, ExperimentMethod
from .contracts import ExperienceRecord, Gripper, Meta, group_by_object, save_experiences
from .evaluation import EvalRow, compute_metrics
from .experiments import EXPERIMENT_CATALOG, experiment_display_name
from .perception import Description, describe
from .pipeline import Pipeline, PipelineRunResult, QueryInput
from .retrieval import build_embedding_text, get_embedding_provider

BASE_URL = "https://raw.githubusercontent.com/expforcesubmission/Exp-Force-Website/main"
SOURCE_RELATIVE = Path("data/expforce/dataset_2gripper.csv")
EXPERIENCES_RELATIVE = Path("data/expforce/validation_experiences.jsonl")
PREPARATION_RELATIVE = Path("data/expforce/preparation_manifest.json")
DESCRIPTORS_RELATIVE = Path("data/expforce/descriptors")
RESULTS_RELATIVE = Path("data/expforce/results")
RUNS_RELATIVE = Path("data/expforce/runs")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"expected True/False, got {value!r}")
    return normalized == "true"


class ExpForceRow(BaseModel):
    object_name: str
    image_name: str
    mass_g: float = Field(gt=0)
    roughness_class: int = Field(ge=1, le=5)
    projected_contact_fraction: float = Field(ge=0, le=1)
    silicone_force_n: float | None
    silicone_feasible: bool
    gecko_force_n: float | None
    gecko_feasible: bool
    favored_gripper: str

    @property
    def object_id(self) -> str:
        return slug(self.object_name)

    @model_validator(mode="after")
    def _validate_labels(self) -> ExpForceRow:
        for force, feasible, name in (
            (self.silicone_force_n, self.silicone_feasible, "silicone"),
            (self.gecko_force_n, self.gecko_feasible, "gecko"),
        ):
            if feasible and (force is None or force <= 0):
                raise ValueError(f"{name} feasible row requires positive force")
            if not feasible and force is not None:
                raise ValueError(f"{name} infeasible row must not contain force")
        expected = self.expected_favored()
        if expected == "tie":
            raise ValueError("synthetic validation rows require one strict winning gripper")
        if self.favored_gripper != expected:
            raise ValueError(
                f"favored_gripper={self.favored_gripper!r}, expected {expected!r}"
            )
        return self

    def expected_favored(self) -> str:
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
    return cfg.root / SOURCE_RELATIVE


def source_sha256(cfg: Config) -> str:
    return hashlib.sha256(source_path(cfg).read_bytes()).hexdigest()


def load_rows(cfg: Config) -> list[ExpForceRow]:
    path = source_path(cfg)
    with path.open(newline="", encoding="utf-8") as fh:
        raw_rows = list(csv.DictReader(fh))
    rows = [
        ExpForceRow(
            object_name=row["Object"],
            image_name=row["Image"],
            mass_g=float(row["Mass_g"]),
            roughness_class=int(row["roughness_class"]),
            projected_contact_fraction=float(row["projected_contact_fraction"]),
            silicone_force_n=float(row["silicone_force_n"]) if row["silicone_force_n"] else None,
            silicone_feasible=_parse_bool(row["silicone_feasible"]),
            gecko_force_n=float(row["gecko_force_n"]) if row["gecko_force_n"] else None,
            gecko_feasible=_parse_bool(row["gecko_feasible"]),
            favored_gripper=row["favored_gripper"].strip().lower(),
        )
        for row in raw_rows
    ]
    ids = [row.object_id for row in rows]
    if len(rows) != 129:
        raise ValueError(f"expected 129 objects, found {len(rows)}")
    if len(set(ids)) != len(ids):
        raise ValueError("object names do not produce unique IDs")
    return rows


def validation_summary(cfg: Config, rows: list[ExpForceRow] | None = None) -> dict:
    rows = rows or load_rows(cfg)
    return {
        "objects": len(rows),
        "experience_rows": 2 * len(rows),
        "source_sha256": source_sha256(cfg),
        "roughness_counts": dict(sorted(Counter(row.roughness_class for row in rows).items())),
        "favored_counts": dict(sorted(Counter(row.favored_gripper for row in rows).items())),
    }


def to_experiences(
    cfg: Config,
    rows: list[ExpForceRow],
    descriptions: dict[str, str] | None = None,
) -> list[ExperienceRecord]:
    descriptions = descriptions or {}
    records: list[ExperienceRecord] = []
    for row in rows:
        image_path = f"data/expforce/images/{row.image_name}"
        description = descriptions.get(row.object_id, row.object_name)
        for gripper, force, feasible in (
            (Gripper.SILICONE, row.silicone_force_n, row.silicone_feasible),
            (Gripper.GECKO, row.gecko_force_n, row.gecko_feasible),
        ):
            records.append(
                ExperienceRecord(
                    object_id=row.object_id,
                    image_path=image_path,
                    mass_g=row.mass_g,
                    roughness_class=row.roughness_class,
                    projected_contact_fraction=row.projected_contact_fraction,
                    gripper=gripper,
                    min_force_n=force if feasible else None,
                    feasible=feasible,
                    failed_at_limit_n=None if feasible else cfg.force.limit_n,
                    semantic_description=description,
                    meta=Meta(pad_id="synthetic-2gripper"),
                )
            )
    return records


class PreparedDescriptor(BaseModel):
    schema_version: int = 2
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
    embedding_sha256: str | None = None
    updated_at: str


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
        temporary = Path(fh.name)
    temporary.replace(path)


def _descriptor_signature(cfg: Config) -> str:
    payload = {
        "model": cfg.models.vlm,
        "system": cfg.prompts.descriptor_system,
        "instruction": cfg.prompts.descriptor,
        "schema": Description.model_json_schema(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _descriptor_path(cfg: Config, object_id: str) -> Path:
    return cfg.root / DESCRIPTORS_RELATIVE / f"{object_id}.json"


def load_prepared_descriptors(cfg: Config) -> dict[str, PreparedDescriptor]:
    root = cfg.root / DESCRIPTORS_RELATIVE
    if not root.exists():
        return {}
    output: dict[str, PreparedDescriptor] = {}
    for path in sorted(root.glob("*.json")):
        try:
            item = PreparedDescriptor.model_validate_json(path.read_text())
        except (OSError, ValueError):
            continue
        output[item.object_id] = item
    return output


def _fallback_description(row: ExpForceRow, reason: str) -> Description:
    return Description(
        retrieval_description=row.object_name,
        contact_region="centered lateral grasp band",
        contact_material="unknown",
        visible_surface_material="unknown",
        visible_surface_condition="unknown",
        local_geometry="unknown",
        contact_patch_visibility=reason,
        uncertainty="No Gemini image descriptor is available.",
    )


def _download_image(image_name: str, destination: Path) -> bool:
    if destination.exists():
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as fh:
        temporary = Path(fh.name)
    try:
        urllib.request.urlretrieve(f"{BASE_URL}/images/{image_name}", temporary)
    except Exception:  # noqa: BLE001 - caller reports every missing image
        temporary.unlink(missing_ok=True)
        return False
    temporary.replace(destination)
    return True


def prepare_dataset(
    cfg: Config,
    *,
    live: bool,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    rows = load_rows(cfg)
    prepared = load_prepared_descriptors(cfg)
    missing_images: list[str] = []
    image_root = cfg.root / "data/expforce/images"
    run_cfg = cfg.model_copy(deep=True)
    run_cfg.models.dry_run = not live
    signature = _descriptor_signature(run_cfg)
    manifest_path = cfg.root / PREPARATION_RELATIVE
    total_steps = len(rows) * (2 if live else 1)
    completed_steps = 0
    descriptors_completed = 0
    embeddings_completed = 0
    manifest = {
        "schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "source_sha256": source_sha256(cfg),
        "descriptor_signature": signature,
        "descriptor_mode": "live_gemini" if live else "offline_or_checkpoint",
        "objects": len(rows),
        "experience_rows": 2 * len(rows),
        "descriptors_completed": 0,
        "embeddings_completed": 0,
        "missing_images": missing_images,
        "failed_stage": None,
        "failed_object": None,
        "error": None,
    }

    def save_manifest() -> None:
        manifest["updated_at"] = datetime.now(UTC).isoformat()
        manifest["missing_images"] = list(missing_images)
        _write_json_atomic(manifest_path, manifest)

    def report(name: str) -> None:
        nonlocal completed_steps
        completed_steps += 1
        if progress is not None:
            progress(completed_steps, total_steps, name)

    for row in rows:
        descriptors_completed += 1
        destination = image_root / row.image_name
        available = destination.exists() or (live and _download_image(row.image_name, destination))
        image_hash = _file_sha256(destination) if available else None
        existing = prepared.get(row.object_id)
        reusable_live = bool(
            existing
            and existing.descriptor_source == "live_gemini"
            and existing.descriptor_signature == signature
            and existing.image_sha256 == image_hash
        )
        if not available:
            missing_images.append(row.image_name)
            descriptor = _fallback_description(row, "image unavailable")
            source = "missing_image_fallback"
        elif reusable_live and existing is not None:
            descriptor = existing.descriptor
            source = existing.descriptor_source
        elif live:
            try:
                import cv2

                image = cv2.imread(str(destination))
                if image is None:
                    raise ValueError(f"could not decode {destination}")
                descriptor = describe(image, run_cfg)
                source = "live_gemini"
            except Exception as error:  # noqa: BLE001 - checkpoint progress before surfacing API failure
                manifest.update(
                    status="failed",
                    failed_stage="descriptor",
                    failed_object=row.object_id,
                    error=f"{type(error).__name__}: {error}",
                )
                save_manifest()
                raise
        elif existing is not None and existing.descriptor_source == "live_gemini":
            descriptor = existing.descriptor
            source = existing.descriptor_source
        else:
            descriptor = _fallback_description(row, "offline object-name fallback")
            source = "object_name_fallback"

        checkpoint = PreparedDescriptor(
            object_id=row.object_id,
            object_name=row.object_name,
            image_name=row.image_name,
            image_path=f"data/expforce/images/{row.image_name}",
            image_sha256=image_hash,
            descriptor_source=source,
            descriptor_model=run_cfg.models.vlm if source == "live_gemini" else None,
            descriptor_signature=signature,
            descriptor=descriptor,
            embedding_status=(existing.embedding_status if reusable_live and existing else "pending"),
            embedding_model=(existing.embedding_model if reusable_live and existing else None),
            embedding_dim=(existing.embedding_dim if reusable_live and existing else None),
            embedding_sha256=(existing.embedding_sha256 if reusable_live and existing else None),
            updated_at=datetime.now(UTC).isoformat(),
        )
        _write_json_atomic(
            _descriptor_path(cfg, row.object_id), checkpoint.model_dump(mode="json")
        )
        prepared[row.object_id] = checkpoint
        manifest["descriptors_completed"] = descriptors_completed
        save_manifest()
        report(f"descriptor: {row.object_name}")

    descriptions = {key: value.descriptor for key, value in prepared.items()}
    records = to_experiences(
        cfg,
        rows,
        {object_id: descriptor.description for object_id, descriptor in descriptions.items()},
    )
    save_experiences(cfg.root / EXPERIENCES_RELATIVE, records)

    if live:
        provider = get_embedding_provider(run_cfg)
        for row in rows:
            embeddings_completed += 1
            checkpoint = prepared[row.object_id]
            try:
                vector = provider.embed(
                    build_embedding_text(checkpoint.descriptor.description),
                    is_query=False,
                )
            except Exception as error:  # noqa: BLE001 - cache keeps every prior successful vector
                manifest.update(
                    status="failed",
                    failed_stage="embedding",
                    failed_object=row.object_id,
                    error=f"{type(error).__name__}: {error}",
                )
                save_manifest()
                raise
            checkpoint.embedding_status = "ready"
            checkpoint.embedding_model = run_cfg.retrieval.embedding.model
            checkpoint.embedding_dim = len(vector)
            checkpoint.embedding_sha256 = hashlib.sha256(vector.tobytes()).hexdigest()
            checkpoint.updated_at = datetime.now(UTC).isoformat()
            _write_json_atomic(
                _descriptor_path(cfg, row.object_id), checkpoint.model_dump(mode="json")
            )
            manifest["embeddings_completed"] = embeddings_completed
            save_manifest()
            report(f"text embedding: {row.object_name}")

    manifest.update(
        status="complete",
        experience_rows=len(records),
        descriptor_source_counts=dict(
            sorted(Counter(item.descriptor_source for item in prepared.values()).items())
        ),
        failed_stage=None,
        failed_object=None,
        error=None,
    )
    save_manifest()
    return manifest


def load_experience_pool(cfg: Config) -> list[ExperienceRecord]:
    path = cfg.root / EXPERIENCES_RELATIVE
    if path.exists():
        from .contracts import load_experiences

        return load_experiences(path)
    return to_experiences(cfg, load_rows(cfg))


def load_validation_records(cfg: Config) -> list[ExperienceRecord]:
    """Compatibility alias for older scripts."""
    return load_experience_pool(cfg)


def load_image(cfg: Config, record: ExperienceRecord):  # noqa: ANN201
    path = cfg.root / record.image_path
    if not path.exists():
        return None
    import cv2

    return cv2.imread(str(path))


@dataclass
class BenchmarkResult:
    metrics: dict
    rows: list[dict]
    run_metadata: dict


def _object_retrieval_payload(detailed: PipelineRunResult) -> list[dict]:
    return [item.model_dump(mode="json") for item in detailed.retrieved_objects]


def pipeline_result_to_dict(detailed: PipelineRunResult) -> dict:
    return {
        "experiment_id": detailed.experiment_id,
        "experiment_method": detailed.experiment_method,
        "experiment_definition_version": detailed.experiment_definition_version,
        "selection": detailed.selection.model_dump(mode="json"),
        "semantic_description": detailed.semantic_description,
        "retrieved_objects": _object_retrieval_payload(detailed),
        "physics_estimates": detailed.physics_estimates,
        "cache_stats": detailed.cache_stats,
    }


def pipeline_result_from_dict(payload: dict) -> PipelineRunResult:
    from .contracts import SelectionResult
    from .retrieval import RetrievedObjectExperience

    return PipelineRunResult(
        experiment_id=payload.get("experiment_id", "legacy"),
        experiment_method=payload.get("experiment_method", "legacy"),
        experiment_definition_version=payload.get("experiment_definition_version", 0),
        selection=SelectionResult.model_validate(payload["selection"]),
        semantic_description=payload.get("semantic_description", ""),
        retrieved_objects=[
            RetrievedObjectExperience.model_validate(item)
            for item in payload.get("retrieved_objects", [])
        ],
        physics_estimates=payload.get("physics_estimates", {}),
        cache_stats=payload.get("cache_stats", {}),
    )


def save_pipeline_run(
    cfg: Config,
    *,
    detailed: PipelineRunResult,
    experiment: str,
    execution_mode: str,
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
        destination = cfg.root / "data/expforce/run_images" / f"{image_digest}.png"
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as fh:
                fh.write(image_bytes)
                temporary = Path(fh.name)
            temporary.replace(destination)
        image_path = str(destination.relative_to(cfg.root))

    run_id = f"{created_at.strftime('%Y%m%dT%H%M%S%fZ')}_{experiment}_{query['object_id']}"
    definition = cfg.experiment(experiment)
    prompt_key = definition.prompt
    prediction_prompts = None
    if prompt_key is not None:
        prediction_prompts = {
            "system": cfg.prompts.prediction_system,
            "instruction_key": prompt_key,
            "instruction": cfg.prompts.experiments[prompt_key],
        }
    artifact = {
        "schema_version": 3,
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
        "prediction_prompts": prediction_prompts,
        "execution_mode": execution_mode,
        "models": {
            "vlm": cfg.models.vlm,
            "embedding": cfg.retrieval.embedding.model,
            "embedding_dim": cfg.retrieval.embedding.dim,
        },
        "inputs": cfg.inputs.model_dump(mode="json"),
        "retrieval_config": cfg.retrieval.model_dump(mode="json"),
        "query": {
            **query,
            "image_artifact_path": image_path,
            "image_sha256": image_digest,
        },
        "counterfactual": counterfactual,
        "truth": truth,
        "result": pipeline_result_to_dict(detailed),
        "baseline": pipeline_result_to_dict(baseline) if baseline is not None else None,
    }
    path = cfg.root / RUNS_RELATIVE / f"{run_id}.json"
    _write_json_atomic(path, artifact)
    return path


def load_saved_runs(cfg: Config) -> list[dict]:
    root = cfg.root / RUNS_RELATIVE
    if not root.exists():
        return []
    runs: list[dict] = []
    for path in sorted(root.glob("*.json"), reverse=True):
        try:
            run = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        run.setdefault("experiment_method", _legacy_experiment_method(run))
        run.setdefault("experiment_definition_version", 0)
        run["artifact_path"] = str(path)
        run["experiment_display_name"] = saved_run_experiment_label(run)
        runs.append(run)
    return runs


def _legacy_experiment_method(run: dict) -> str:
    toggles = run.get("experiment_toggles", {})
    if toggles.get("use_residual"):
        return ExperimentMethod.PHYSICS_SEMANTIC_RESIDUAL.value
    if toggles.get("use_physics"):
        return ExperimentMethod.CALIBRATED_PHYSICS.value
    if (
        toggles.get("use_paired_rows")
        and toggles.get("use_retrieval")
        and toggles.get("use_vlm")
    ):
        return ExperimentMethod.PAIRED_RETRIEVAL_VLM.value
    if toggles.get("use_vlm"):
        return "legacy_per_gripper_vlm"
    if toggles.get("use_retrieval"):
        return "legacy_branch_retrieval_average"
    return "legacy_unknown"


def saved_run_experiment_label(run: dict) -> str:
    """Human-readable provenance that prevents old E4/E5 IDs from being misread."""
    experiment = str(run.get("experiment", "unknown")).lower()
    version = int(run.get("experiment_definition_version", 0) or 0)
    if version >= EXPERIMENT_DEFINITION_VERSION and experiment in EXPERIMENT_CATALOG:
        return experiment_display_name(experiment)

    method = str(run.get("experiment_method") or _legacy_experiment_method(run))
    legacy_labels = {
        ExperimentMethod.PAIRED_RETRIEVAL_VLM.value: "paired retrieval VLM",
        ExperimentMethod.CALIBRATED_PHYSICS.value: "calibrated physics",
        ExperimentMethod.PHYSICS_SEMANTIC_RESIDUAL.value: "physics + semantic residual",
        "legacy_per_gripper_vlm": "per-gripper VLM",
        "legacy_branch_retrieval_average": "branch retrieval average",
        "legacy_unknown": "unknown method",
    }
    return f"Legacy {experiment.upper()} — {legacy_labels.get(method, method)}"


def run_benchmark(
    cfg: Config,
    experiment: str,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> BenchmarkResult:
    records = load_experience_pool(cfg)
    by_object = group_by_object(records)
    object_ids = sorted(by_object)
    eval_rows: list[EvalRow] = []
    output_rows: list[dict] = []

    for index, object_id in enumerate(object_ids, start=1):
        train = [record for record in records if record.object_id != object_id]
        pipe = Pipeline(cfg, experiment).fit(train)
        truth = by_object[object_id]
        sample = truth.gecko or truth.silicone
        assert sample is not None
        query = QueryInput(
            object_id=object_id,
            mass_g=sample.mass_g,
            roughness_class=sample.roughness_class,
            projected_contact_fraction=sample.projected_contact_fraction,
            image_bgr=load_image(cfg, sample),
            image_path=sample.image_path,
            semantic_description=sample.semantic_description,
        )
        detailed = pipe.predict_detailed(query)
        result = detailed.selection
        eval_rows.append(EvalRow(object_id=object_id, truth=truth, result=result))
        optimal, oracle_force = truth.optimal_grippers()
        if len(optimal) != 1:
            raise ValueError(f"validation object {object_id!r} does not have a strict winner")
        chosen = result.desired_gripper
        chosen_truth = truth.get(Gripper(chosen)) if chosen in {"gecko", "silicone"} else None
        regret = None
        if oracle_force is not None and chosen_truth and chosen_truth.feasible:
            regret = (chosen_truth.min_force_n or 0.0) - oracle_force
        row = {
            "object_id": object_id,
            "mass_g": sample.mass_g,
            "roughness_class": sample.roughness_class,
            "projected_contact_fraction": sample.projected_contact_fraction,
            "true_gecko_force_n": truth.gecko.min_force_n if truth.gecko else None,
            "pred_gecko_force_n": result.candidate_predictions["gecko"].predicted_normal_force_n,
            "true_silicone_force_n": truth.silicone.min_force_n if truth.silicone else None,
            "pred_silicone_force_n": result.candidate_predictions["silicone"].predicted_normal_force_n,
            "true_favored": next(iter(optimal)).value,
            "predicted_gripper": chosen,
            "selection_correct": chosen in {gripper.value for gripper in optimal},
            "regret_n": regret,
            "model_recommended_gripper": result.model_recommended_gripper,
            "recommendation_agrees_with_selector": result.recommendation_agrees_with_selector,
            "semantic_description": detailed.semantic_description,
            "retrieved_objects": _object_retrieval_payload(detailed),
            "physics_estimates": detailed.physics_estimates,
            "cache_stats": detailed.cache_stats,
        }
        output_rows.append(row)
        if progress is not None:
            progress(index, len(object_ids), object_id)

    metrics = compute_metrics(eval_rows, cfg).to_dict()
    definition = cfg.experiment(experiment)
    prompt_key = definition.prompt
    prediction_prompts = None
    if prompt_key is not None:
        prediction_prompts = {
            "system": cfg.prompts.prediction_system,
            "instruction_key": prompt_key,
            "instruction": cfg.prompts.experiments[prompt_key],
        }
    metadata = {
        "schema_version": 3,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": experiment,
        "experiment_method": definition.method.value,
        "experiment_definition_version": EXPERIMENT_DEFINITION_VERSION,
        "experiment_definition": definition.model_dump(mode="json"),
        "dry_run": cfg.models.dry_run,
        "source_sha256": source_sha256(cfg),
        "evaluation_protocol": "leave-one-object-out",
        "experience_pool_objects": len(object_ids),
        "training_objects_per_run": len(object_ids) - 1,
        "model": cfg.models.vlm,
        "embedding_model": cfg.retrieval.embedding.model,
        "embedding_dim": cfg.retrieval.embedding.dim,
        "inputs": cfg.inputs.model_dump(mode="json"),
        "retrieval": cfg.retrieval.model_dump(mode="json"),
        "prediction_prompts": prediction_prompts,
    }
    return BenchmarkResult(metrics=metrics, rows=output_rows, run_metadata=metadata)


def save_benchmark(cfg: Config, benchmark: BenchmarkResult) -> tuple[Path, Path]:
    output_dir = cfg.root / RESULTS_RELATIVE
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{timestamp}_{benchmark.run_metadata['experiment']}"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    json_path.write_text(
        json.dumps(
            {
                "metadata": benchmark.run_metadata,
                "metrics": benchmark.metrics,
                "rows": benchmark.rows,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    flat_fields = [
        key
        for key in benchmark.rows[0]
        if key not in {"retrieved_objects", "cache_stats"}
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=flat_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(benchmark.rows)
    return json_path, csv_path
