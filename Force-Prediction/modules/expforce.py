"""Exp-Force experience-pool preparation, inspection, and benchmark tools."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from .config import (
    EXPERIMENT_DEFINITION_VERSION,
    Config,
    ExperimentMethod,
    prompt_bundle_sha256,
)
from .contracts import ExperienceRecord, Gripper, Meta, group_by_object
from .evaluation import EvalRow, compute_metrics
from .experiments import EXPERIMENT_CATALOG, experiment_display_name
from .pipeline import Pipeline, PipelineRunResult, QueryInput

BASE_URL = "https://raw.githubusercontent.com/expforcesubmission/Exp-Force-Website/main"
SOURCE_RELATIVE = Path("data/expforce/dataset.csv")
EXPERIENCES_RELATIVE = Path("data/cache/expforce/experiences.jsonl")
PREPARATION_RELATIVE = Path("data/expforce/preparation_manifest.json")
DESCRIPTORS_RELATIVE = Path("data/expforce/objects")
RESULTS_RELATIVE = Path("data/expforce/results")
RUNS_RELATIVE = Path("data/expforce/runs")
SUITES_RELATIVE = Path("data/expforce/suites")


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


def _parse_optional_int(value: str | None) -> int | None:
    return int(value) if value and value.strip() else None


class ExpForceRow(BaseModel):
    object_name: str
    image_name: str
    image_name_2: str | None = None
    mass_g: float | None = Field(default=None, gt=0)
    roughness_class: int | None = Field(default=None, ge=1, le=5)
    projected_contact_fraction: float | None = Field(default=None, ge=0, le=1)
    silicone_force_n: float | None = Field(default=None, gt=0)
    silicone_feasible: bool | None = None
    gecko_force_n: float | None = Field(default=None, gt=0)
    gecko_feasible: bool | None = None
    favored_gripper: str | None = None

    @property
    def object_id(self) -> str:
        return slug(self.object_name)

    @model_validator(mode="after")
    def _validate_labels(self) -> ExpForceRow:
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


def _dataset_root(cfg: Config) -> Path:
    return cfg.root / "data" / cfg.dataset_id


def _dataset_relative(cfg: Config, name: str) -> Path:
    return Path("data") / cfg.dataset_id / name


def _object_image_relative(cfg: Config, row: ExpForceRow) -> Path:
    suffix = Path(row.image_name).suffix.lower() or ".png"
    return _dataset_relative(cfg, f"objects/{row.object_id}/image{suffix}")


def _experience_cache_path(cfg: Config) -> Path:
    return cfg.root / "data" / "cache" / cfg.dataset_id / "experiences.jsonl"


def source_path(cfg: Config) -> Path:
    return _dataset_root(cfg) / "dataset.csv"


def source_sha256(cfg: Config) -> str:
    path = source_path(cfg)
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    from .datasets.catalog import get_dataset

    return get_dataset(cfg, cfg.dataset_id).source_fingerprint


def load_rows(cfg: Config) -> list[ExpForceRow]:
    path = source_path(cfg)
    with path.open(newline="", encoding="utf-8") as fh:
        raw_rows = list(csv.DictReader(fh))
    rows = [
        ExpForceRow(
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
            mass_g=_parse_optional_float(row.get("Mass_g")),
            roughness_class=_parse_optional_int(row.get("roughness_class")),
            projected_contact_fraction=_parse_optional_float(
                row.get("projected_contact_fraction")
            ),
            silicone_force_n=_parse_optional_float(row.get("silicone_force_n")),
            silicone_feasible=_parse_optional_bool(row.get("silicone_feasible")),
            gecko_force_n=_parse_optional_float(row.get("gecko_force_n")),
            gecko_feasible=_parse_optional_bool(row.get("gecko_feasible")),
            favored_gripper=(
                row.get("favored_gripper", "").strip().lower() or None
            ),
        )
        for row in raw_rows
    ]
    ids = [row.object_id for row in rows]
    if cfg.dataset_id == "expforce" and len(rows) != 129:
        raise ValueError(f"expected 129 objects, found {len(rows)}")
    if len(set(ids)) != len(ids):
        raise ValueError("object names do not produce unique IDs")
    return rows


def save_rows(path: Path, rows: list[ExpForceRow]) -> None:
    """Atomically persist validated paired-object source rows."""
    object_ids = [row.object_id for row in rows]
    if len(set(object_ids)) != len(object_ids):
        raise ValueError("object names do not produce unique IDs")

    columns = [
        "Object",
        "Image",
        *(["Image_2"] if any(row.image_name_2 for row in rows) else []),
        "Mass_g",
        "roughness_class",
        "projected_contact_fraction",
        "silicone_force_n",
        "silicone_feasible",
        "gecko_force_n",
        "gecko_feasible",
        "favored_gripper",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        newline="",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            payload = {
                "Object": row.object_name,
                "Image": row.image_name,
                "Mass_g": row.mass_g if row.mass_g is not None else "",
                "roughness_class": (
                    row.roughness_class if row.roughness_class is not None else ""
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


def validation_summary(cfg: Config, rows: list[ExpForceRow] | None = None) -> dict:
    rows = rows or load_rows(cfg)
    return {
        "objects": len(rows),
        "experience_rows": sum(
            (row.gecko_feasible is False or (
                row.gecko_feasible is True and row.gecko_force_n is not None
            ))
            + (row.silicone_feasible is False or (
                row.silicone_feasible is True and row.silicone_force_n is not None
            ))
            for row in rows
        ),
        "source_sha256": source_sha256(cfg),
        "roughness_counts": dict(sorted(Counter(
            row.roughness_class for row in rows if row.roughness_class is not None
        ).items())),
        "favored_counts": dict(sorted(Counter(
            row.favored_gripper for row in rows if row.favored_gripper is not None
        ).items())),
    }


def to_experiences(
    cfg: Config,
    rows: list[ExpForceRow],
    descriptions: dict[str, str] | None = None,
) -> list[ExperienceRecord]:
    descriptions = descriptions or {}
    records: list[ExperienceRecord] = []
    for row in rows:
        image_path = str(_object_image_relative(cfg, row))
        description = descriptions.get(row.object_id, row.object_name)
        for gripper, force, feasible in (
            (Gripper.SILICONE, row.silicone_force_n, row.silicone_feasible),
            (Gripper.GECKO, row.gecko_force_n, row.gecko_feasible),
        ):
            if feasible is None or (feasible and force is None):
                continue
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
                    meta=Meta(
                        pad_id="synthetic-2gripper",
                        contact_fraction_source="synthetic_fixture",
                    ),
                )
            )
    return records


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
        temporary = Path(fh.name)
    temporary.replace(path)


def prepare_dataset(
    cfg: Config,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Compatibility wrapper around the canonical Gemini preparation pipeline."""
    from .datasets import PreparationStage, get_dataset, prepare_dataset_stages

    dataset = get_dataset(cfg, cfg.dataset_id)
    return prepare_dataset_stages(
        cfg,
        dataset,
        [
            PreparationStage.DESCRIPTIONS,
            PreparationStage.EMBEDDINGS,
            PreparationStage.EXPERIENCES,
        ],
        progress=progress,
    )


def load_experience_pool(cfg: Config) -> list[ExperienceRecord]:
    path = _experience_cache_path(cfg)
    if path.exists():
        from .contracts import load_experiences

        return load_experiences(path)
    if source_path(cfg).is_file():
        return to_experiences(cfg, load_rows(cfg))
    from .datasets import get_dataset
    from .datasets.storage import dataset_experience_records

    dataset = get_dataset(cfg, cfg.dataset_id)
    return dataset_experience_records(dataset, cfg.force.limit_n)


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
        "retrieval_mode": detailed.retrieval_mode,
        "effective_inputs": list(detailed.effective_inputs),
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
        retrieval_mode=payload.get("retrieval_mode"),
        effective_inputs=tuple(payload.get("effective_inputs", ())),
    )


def prompt_provenance(cfg: Config, prompt_key: str | None) -> dict:
    """Exact prompt and fixed embodiment context used by a saved run."""
    embodiments = {
        name: {"description": context.description}
        for name, context in cfg.embodiments.items()
    }
    prediction = None
    if prompt_key is not None:
        prediction = {
            "system": cfg.prompts.prediction_system,
            "instruction_key": prompt_key,
            "instruction": cfg.prompts.experiments[prompt_key],
        }
    return {
        "bundle_file": cfg.prompts_file,
        "bundle_sha256": prompt_bundle_sha256(cfg),
        "experiment_instructions": dict(cfg.prompts.experiments),
        "prediction": prediction,
        "descriptor": {
            "system": cfg.prompts.descriptor_system,
            "instruction": cfg.prompts.descriptor,
        },
        "embodiments": embodiments,
    }


def backend_provenance(cfg: Config, experiment: str) -> dict[str, str | None]:
    """Record the actual force and semantic backends used by an experiment."""
    if experiment in {"e1", "e2", "e3", "e4"}:
        return {
            "force": "gemini_joint_generation",
            "semantic_embedding": (
                cfg.retrieval.embedding.model if experiment in {"e3", "e4"} else None
            ),
        }
    raise KeyError(f"unknown experiment {experiment!r}")


def artifact_backend_label(payload: dict) -> str:
    """Human-readable backend label, including read-only legacy provenance."""
    backend = payload.get("backend") or payload.get("metadata", {}).get("backend")
    if backend:
        force = backend.get("force", "unknown")
        embedding = backend.get("semantic_embedding")
        return f"{force} + {embedding}" if embedding else force
    if "execution_mode" in payload:
        return f"Legacy {payload['execution_mode']}"
    metadata = payload.get("metadata", payload)
    if "dry_run" in metadata:
        return "Legacy Offline" if metadata["dry_run"] else "Legacy Live Gemini"
    return "Unknown"


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
    prompt_key = definition.prompt
    prompt_context = prompt_provenance(cfg, prompt_key)
    artifact = {
        "schema_version": 7,
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
        run.setdefault("experiment_method", _legacy_experiment_method(run))
        run.setdefault("experiment_definition_version", 0)
        run["artifact_path"] = str(path)
        run["experiment_display_name"] = saved_run_experiment_label(run)
        run["backend_label"] = artifact_backend_label(run)
        runs.append(run)
    return runs


def _legacy_experiment_method(run: dict) -> str:
    toggles = run.get("experiment_toggles", {})
    if toggles.get("use_residual"):
        return "physics_semantic_residual"
    if toggles.get("use_physics"):
        return "calibrated_physics"
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
    method = str(run.get("experiment_method") or _legacy_experiment_method(run))
    if experiment in EXPERIMENT_CATALOG:
        expected = EXPERIMENT_CATALOG[experiment].method.value
        if version >= 3 and method == expected:
            label = experiment_display_name(experiment)
            return label if version == EXPERIMENT_DEFINITION_VERSION else f"{label} (v{version})"

    legacy_labels = {
        ExperimentMethod.SEMANTIC_RETRIEVAL_VLM.value: "semantic experiential retrieval",
        ExperimentMethod.PAIRED_RETRIEVAL_VLM.value: "paired retrieval VLM",
        "calibrated_physics": "calibrated physics",
        "physics_semantic_residual": "physics + semantic residual",
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
    from .datasets import get_dataset
    from .experiments import experiment_eligibility

    dataset = get_dataset(cfg, cfg.dataset_id)
    eligibility = experiment_eligibility(dataset, cfg, experiment)
    records = load_experience_pool(cfg)
    by_object = group_by_object(records)
    object_ids = list(eligibility.benchmark_ids)
    eval_rows: list[EvalRow] = []
    output_rows: list[dict] = []

    for index, object_id in enumerate(object_ids, start=1):
        train = [
            record
            for record in records
            if record.object_id != object_id
            and (
                experiment not in {"e3", "e4"}
                or record.object_id in eligibility.reference_ids
            )
        ]
        pipe = Pipeline(cfg, experiment).fit(train)
        truth = by_object[object_id]
        item = dataset.objects[object_id]
        image_path = cfg.root / item.image.path
        image = None
        if image_path.is_file():
            import cv2

            image = cv2.imread(str(image_path))
        query = QueryInput(
            object_id=object_id,
            mass_g=item.mass_g,
            roughness_class=item.roughness_class,
            projected_contact_fraction=item.projected_contact_fraction,
            image_bgr=image,
            image_path=item.image.path,
            semantic_description=(
                item.description.value.description if item.description is not None else None
            ),
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
            "mass_g": item.mass_g,
            "roughness_class": item.roughness_class,
            "projected_contact_fraction": item.projected_contact_fraction,
            "true_gecko_force_n": truth.gecko.min_force_n if truth.gecko else None,
            "true_gecko_feasible": truth.gecko.feasible if truth.gecko else None,
            "pred_gecko_force_n": result.candidate_predictions["gecko"].predicted_normal_force_n,
            "pred_gecko_feasible": result.candidate_predictions["gecko"].feasible,
            "true_silicone_force_n": truth.silicone.min_force_n if truth.silicone else None,
            "true_silicone_feasible": truth.silicone.feasible if truth.silicone else None,
            "pred_silicone_force_n": result.candidate_predictions["silicone"].predicted_normal_force_n,
            "pred_silicone_feasible": result.candidate_predictions["silicone"].feasible,
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
    prompt_context = prompt_provenance(cfg, prompt_key)
    retrieval_mode = EXPERIMENT_CATALOG[experiment].retrieval_mode
    metadata = {
        "schema_version": 7,
        "dataset_id": cfg.dataset_id,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": experiment,
        "experiment_method": definition.method.value,
        "experiment_definition_version": EXPERIMENT_DEFINITION_VERSION,
        "experiment_definition": definition.model_dump(mode="json"),
        "backend": backend_provenance(cfg, experiment),
        "source_sha256": source_sha256(cfg),
        "evaluation_protocol": "leave-one-object-out",
        "experience_pool_objects": len(object_ids),
        "training_objects_per_run": max(0, len(eligibility.reference_ids) - 1),
        "model": cfg.models.vlm,
        "embedding_model": cfg.retrieval.embedding.model,
        "embedding_dim": cfg.retrieval.embedding.dim,
        "inputs": cfg.inputs.model_dump(mode="json"),
        "retrieval": cfg.retrieval.model_dump(mode="json"),
        "prediction_prompts": prompt_context["prediction"],
        "prompt_context": prompt_context,
        "retrieval_mode": retrieval_mode.value if retrieval_mode is not None else None,
        "eligibility": {
            "eligible_query_ids": list(eligibility.query_ids),
            "eligible_benchmark_ids": list(eligibility.benchmark_ids),
            "reference_ids": list(eligibility.reference_ids),
            "skipped_queries": eligibility.skipped_queries,
            "skipped_benchmarks": eligibility.skipped_benchmarks,
        },
    }
    return BenchmarkResult(metrics=metrics, rows=output_rows, run_metadata=metadata)


def save_benchmark(cfg: Config, benchmark: BenchmarkResult) -> tuple[Path, Path]:
    output_dir = _dataset_root(cfg) / "results"
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
