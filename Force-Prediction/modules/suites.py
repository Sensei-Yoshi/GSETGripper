"""Resumable two-stage E1–E6 prediction and evaluation suites."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path

from .benchmarking import (
    BenchmarkEvaluation,
    BenchmarkPredictionBatch,
    evaluation_readiness,
    generate_benchmark_predictions,
    get_or_create_benchmark_evaluation,
    load_benchmark_evaluation,
    load_prediction_batch,
    save_prediction_batch,
)
from .config import EXPERIMENT_DEFINITION_VERSION, Config, prompt_bundle_sha256
from .datasets.storage import write_json_atomic
from .experiments import EXPERIMENT_CATALOG, experiment_eligibility
from .expforce import prompt_provenance
from .reporting import (
    common_intersection_artifacts,
    export_comparison,
)

PRIMARY_EXPERIMENTS = ("e1", "e2", "e3", "e4", "e5", "e6")
SUITE_SCHEMA_VERSION = 11
SUITE_REPORTING_VERSION = 2


def normalize_suite_experiments(
    experiments: Iterable[str] | None,
) -> tuple[str, ...]:
    """Resolve a requested subset into canonical order, defaulting to all primaries."""
    if experiments is None:
        return PRIMARY_EXPERIMENTS
    selected = {name.lower() for name in experiments}
    unknown = selected - set(PRIMARY_EXPERIMENTS)
    if unknown:
        raise ValueError(f"unknown suite experiments: {sorted(unknown)}")
    ordered = tuple(name for name in PRIMARY_EXPERIMENTS if name in selected)
    if not ordered:
        raise ValueError("a suite needs at least one experiment")
    return ordered


def suite_experiments(manifest: dict) -> tuple[str, ...]:
    """The experiments a manifest runs, tolerant of legacy manifests without the field."""
    return normalize_suite_experiments(manifest.get("experiments"))


def _definition_snapshot(cfg: Config, experiments: tuple[str, ...]) -> dict:
    """Configuration lock that intentionally excludes mutable dataset truth."""
    from .datasets import get_dataset

    dataset = get_dataset(cfg, cfg.dataset_id)
    split = {
        "train": sorted(
            item.object_id for item in dataset.objects.values() if item.split == "train"
        ),
        "test": sorted(
            item.object_id for item in dataset.objects.values() if item.split == "test"
        ),
    }
    definitions = {
        name: cfg.experiment(name).model_dump(mode="json")
        for name in experiments
    }
    return {
        "dataset_id": cfg.dataset_id,
        "prompt_bundle_sha256": prompt_bundle_sha256(cfg),
        "experiment_definition_version": EXPERIMENT_DEFINITION_VERSION,
        "experiment_definitions": definitions,
        "experiment_input_profiles": {
            name: EXPERIMENT_CATALOG[name].scoped_config(cfg).inputs.model_dump(mode="json")
            for name in experiments
        },
        "models": {
            "vlm": cfg.models.vlm,
            "embedding": cfg.retrieval.embedding.model,
            "embedding_dim": cfg.retrieval.embedding.dim,
        },
        "backend": (
            "gemini_single_generation"
            if len(cfg.prediction.active_grippers) == 1
            else "gemini_joint_generation"
        ),
        "active_grippers": [
            gripper.value for gripper in cfg.prediction.active_grippers
        ],
        "generation_mode": (
            "single" if len(cfg.prediction.active_grippers) == 1 else "joint"
        ),
        "split": split,
        "split_sha256": _snapshot_hash(split),
        "retrieval": cfg.retrieval.model_dump(mode="json"),
        "roughness_measurement": cfg.roughness.model_dump(mode="json"),
        "inputs": cfg.inputs.model_dump(mode="json"),
    }


def _snapshot_hash(snapshot: dict) -> str:
    encoded = json.dumps(snapshot, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def suite_manifest_path(cfg: Config, suite_id: str) -> Path:
    return cfg.root / "data" / cfg.dataset_id / "suites" / suite_id / "manifest.json"


def create_suite(
    cfg: Config,
    experiments: Iterable[str] | None = None,
) -> dict:
    selected = normalize_suite_experiments(experiments)
    created_at = datetime.now(UTC)
    suite_id = created_at.strftime("%Y%m%dT%H%M%S%fZ_") + "_".join(selected)
    snapshot = _definition_snapshot(cfg, selected)
    manifest = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "suite_id": suite_id,
        "created_at": created_at.isoformat(),
        "updated_at": created_at.isoformat(),
        "status": "pending",
        "backend": snapshot["backend"],
        "active_grippers": snapshot["active_grippers"],
        "generation_mode": snapshot["generation_mode"],
        "experiments": list(selected),
        "definition_snapshot": snapshot,
        "definition_snapshot_sha256": _snapshot_hash(snapshot),
        "prompt_context": prompt_provenance(cfg, selected[0]),
        "runs": {
            name: {
                "status": "pending",
                "prediction_json_path": None,
                "prediction_csv_path": None,
            }
            for name in selected
        },
        "evaluations": [],
    }
    write_json_atomic(suite_manifest_path(cfg, suite_id), manifest)
    return manifest


def load_suite(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def list_suites(cfg: Config) -> list[dict]:
    root = cfg.root / "data" / cfg.dataset_id / "suites"
    manifests: list[dict] = []
    for path in sorted(root.glob("*/manifest.json"), reverse=True):
        try:
            manifest = load_suite(path)
        except (OSError, json.JSONDecodeError):
            continue
        manifest["manifest_path"] = str(path)
        manifests.append(manifest)
    return manifests


def _validate_resumable(cfg: Config, manifest: dict) -> None:
    if manifest.get("schema_version") != SUITE_SCHEMA_VERSION:
        raise ValueError(
            f"Legacy suites are read-only; start a schema-v{SUITE_SCHEMA_VERSION} "
            "suite for two-stage execution"
        )
    current = _definition_snapshot(cfg, suite_experiments(manifest))
    if _snapshot_hash(current) != manifest["definition_snapshot_sha256"]:
        raise ValueError(
            "Current prompts, model, retrieval, inputs, active grippers, or experiment "
            "definitions differ from this suite; start a new suite instead"
        )


def _suite_status(manifest: dict) -> str:
    statuses = [state["status"] for state in manifest["runs"].values()]
    if all(status == "completed" for status in statuses):
        return "predictions_complete"
    if any(status == "completed" for status in statuses):
        return "partially_generated"
    return "waiting"


def run_suite_predictions(
    cfg: Config,
    manifest: dict | None = None,
    *,
    progress: Callable[[str, int, int, str], None] | None = None,
) -> dict:
    """Generate or resume immutable prediction batches for currently ready experiments."""
    manifest = manifest or create_suite(cfg)
    _validate_resumable(cfg, manifest)
    path = suite_manifest_path(cfg, manifest["suite_id"])
    manifest["status"] = "running_predictions"
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    write_json_atomic(path, manifest)

    try:
        from .datasets import get_dataset

        dataset = get_dataset(cfg, cfg.dataset_id)
        for experiment in suite_experiments(manifest):
            state = manifest["runs"][experiment]
            saved = state.get("prediction_json_path")
            if (
                state.get("status") == "completed"
                and saved
                and (cfg.root / saved).is_file()
            ):
                continue

            readiness = experiment_eligibility(dataset, cfg, experiment)
            if not readiness.query_ids:
                state.update(
                    status="waiting",
                    reason="no query-ready objects",
                    query_count=0,
                    skipped_queries=readiness.skipped_queries,
                    checked_at=datetime.now(UTC).isoformat(),
                )
                write_json_atomic(path, manifest)
                continue

            state.update(
                status="running",
                reason=None,
                query_count=len(readiness.query_ids),
            )
            write_json_atomic(path, manifest)

            def on_object(
                done: int,
                total: int,
                object_id: str,
                active_experiment: str = experiment,
            ) -> None:
                if progress is not None:
                    progress(active_experiment, done, total, object_id)

            batch = generate_benchmark_predictions(
                cfg,
                experiment,
                progress=on_object,
            )
            json_path, csv_path = save_prediction_batch(cfg, batch)
            state.update(
                status="completed",
                prediction_batch_id=batch.batch_id,
                prediction_json_path=str(json_path.relative_to(cfg.root)),
                prediction_csv_path=str(csv_path.relative_to(cfg.root)),
                completed_at=datetime.now(UTC).isoformat(),
                query_count=len(batch.rows),
                generation_source_sha256=batch.metadata[
                    "generation_source_sha256"
                ],
                generation_input_sha256=batch.metadata[
                    "generation_input_sha256"
                ],
            )
            manifest["updated_at"] = datetime.now(UTC).isoformat()
            write_json_atomic(path, manifest)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["last_error"] = f"{type(error).__name__}: {error}"
        manifest["updated_at"] = datetime.now(UTC).isoformat()
        write_json_atomic(path, manifest)
        raise

    manifest["status"] = _suite_status(manifest)
    manifest.pop("last_error", None)
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    write_json_atomic(path, manifest)
    return manifest


def run_suite(
    cfg: Config,
    manifest: dict | None = None,
    *,
    progress: Callable[[str, int, int, str], None] | None = None,
) -> dict:
    """Compatibility alias for callers migrating to ``run_suite_predictions``."""
    return run_suite_predictions(cfg, manifest, progress=progress)


def suite_prediction_batches(
    cfg: Config,
    manifest: dict,
) -> dict[str, BenchmarkPredictionBatch]:
    batches: dict[str, BenchmarkPredictionBatch] = {}
    if manifest.get("schema_version") != SUITE_SCHEMA_VERSION:
        return batches
    for experiment in suite_experiments(manifest):
        relative = manifest["runs"][experiment].get("prediction_json_path")
        if relative and (cfg.root / relative).is_file():
            batches[experiment] = load_prediction_batch(cfg.root / relative)
    return batches


def _suite_evaluation_signature(
    evaluations: dict[str, BenchmarkEvaluation],
) -> str:
    payload = {
        experiment: evaluation.metadata["truth_snapshot_sha256"]
        for experiment, evaluation in evaluations.items()
    }
    return _snapshot_hash(payload)


def evaluate_suite(cfg: Config, manifest: dict) -> dict:
    """Evaluate every generated suite batch against one current truth state."""
    if manifest.get("schema_version") != SUITE_SCHEMA_VERSION:
        raise ValueError("Legacy suites are read-only and cannot be reevaluated")
    batches = suite_prediction_batches(cfg, manifest)
    evaluations: dict[str, BenchmarkEvaluation] = {}
    paths_by_experiment: dict[str, dict[str, str]] = {}
    for experiment, batch in batches.items():
        if not evaluation_readiness(cfg, batch).eligible_ids:
            continue
        evaluation, paths, _ = get_or_create_benchmark_evaluation(cfg, batch)
        evaluations[experiment] = evaluation
        paths_by_experiment[experiment] = {
            name: str(path.relative_to(cfg.root)) for name, path in paths.items()
        }
    if not evaluations:
        raise ValueError("no generated suite predictions currently have evaluation-ready truth")

    signature = _suite_evaluation_signature(evaluations)
    for existing in manifest.get("evaluations", []):
        if (
            existing.get("truth_snapshot_sha256") == signature
            and existing.get("reporting_version", 1) == SUITE_REPORTING_VERSION
        ):
            return manifest

    created_at = datetime.now(UTC)
    suite_evaluation_id = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    artifacts = {
        experiment: evaluation.to_artifact()
        for experiment, evaluation in evaluations.items()
    }
    comparable, common_ids = common_intersection_artifacts(artifacts, cfg)
    exports: dict[str, str] = {}
    if common_ids:
        destination = (
            suite_manifest_path(cfg, manifest["suite_id"]).parent
            / "evaluations"
            / suite_evaluation_id
        )
        exported = export_comparison(
            comparable,
            destination,
            image_root=cfg.root,
        )
        exports = {
            name: str(Path(path).relative_to(cfg.root))
            for name, path in exported.items()
        }

    manifest.setdefault("evaluations", []).append(
        {
            "evaluation_id": suite_evaluation_id,
            "created_at": created_at.isoformat(),
            "reporting_version": SUITE_REPORTING_VERSION,
            "truth_snapshot_sha256": signature,
            "artifacts": paths_by_experiment,
            "coverage": {
                experiment: evaluation.metadata["coverage"]
                for experiment, evaluation in evaluations.items()
            },
            "common_object_ids": list(common_ids),
            "exports": exports,
        }
    )
    manifest["status"] = "evaluated"
    manifest["updated_at"] = created_at.isoformat()
    write_json_atomic(suite_manifest_path(cfg, manifest["suite_id"]), manifest)
    return manifest


def suite_evaluation_artifacts(
    cfg: Config,
    manifest: dict,
    suite_evaluation: dict | None = None,
) -> dict[str, dict]:
    """Load one suite evaluation's per-experiment evaluation artifacts."""
    evaluations = manifest.get("evaluations", [])
    selected = suite_evaluation or (evaluations[-1] if evaluations else None)
    if selected is None:
        return {}
    artifacts: dict[str, dict] = {}
    for experiment, paths in selected.get("artifacts", {}).items():
        json_relative = paths.get("json")
        if not json_relative:
            continue
        path = cfg.root / json_relative
        if path.is_file():
            artifacts[experiment] = load_benchmark_evaluation(path).to_artifact()
    return artifacts


def suite_benchmarks(cfg: Config, manifest: dict) -> dict[str, dict]:
    """Compatibility reader: latest v9 evaluations or saved v8 benchmark artifacts."""
    if manifest.get("schema_version") == SUITE_SCHEMA_VERSION:
        return suite_evaluation_artifacts(cfg, manifest)
    artifacts: dict[str, dict] = {}
    for experiment in PRIMARY_EXPERIMENTS:
        relative = manifest.get("runs", {}).get(experiment, {}).get("json_path")
        if not relative:
            continue
        path = cfg.root / relative
        if path.is_file():
            artifacts[experiment] = json.loads(path.read_text(encoding="utf-8"))
    return artifacts


def update_suite_exports(cfg: Config, manifest: dict, exports: dict[str, str]) -> dict:
    """Compatibility helper for manually exported legacy suite comparisons."""
    manifest["exports"] = exports
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    write_json_atomic(suite_manifest_path(cfg, manifest["suite_id"]), manifest)
    return manifest
