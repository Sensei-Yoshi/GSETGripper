"""Two-stage benchmark prediction generation, evaluation, and persistence."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import EXPERIMENT_DEFINITION_VERSION, Config
from .contracts import Gripper, SelectionResult, group_by_object
from .datasets import get_dataset
from .datasets.storage import dataset_experience_records, write_json_atomic
from .evaluation import EvalRow, compute_metrics
from .experiments import (
    EXPERIMENT_CATALOG,
    evaluation_truth_eligibility,
    experiment_eligibility,
)
from .expforce import (
    backend_provenance,
    load_experience_pool,
    pipeline_result_from_dict,
    pipeline_result_to_dict,
    prompt_provenance,
    source_sha256,
)
from .pipeline import Pipeline, QueryInput

BENCHMARK_SCHEMA_VERSION = 10
PREDICTION_ARTIFACT_TYPE = "benchmark_prediction_batch"
EVALUATION_ARTIFACT_TYPE = "benchmark_evaluation"


@dataclass(frozen=True)
class BenchmarkScope:
    """Train/reference and held-out query membership for one experiment."""

    mode: str
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    reference_ids: tuple[str, ...]
    skipped_queries: dict[str, tuple[str, ...]]


@dataclass
class BenchmarkPredictionBatch:
    metadata: dict[str, Any]
    rows: list[dict[str, Any]]
    json_path: Path | None = None
    csv_path: Path | None = None

    @property
    def batch_id(self) -> str:
        return str(self.metadata["batch_id"])

    def to_artifact(self) -> dict[str, Any]:
        return {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "artifact_type": PREDICTION_ARTIFACT_TYPE,
            "metadata": self.metadata,
            "rows": self.rows,
        }


@dataclass
class BenchmarkEvaluation:
    metadata: dict[str, Any]
    metrics: dict[str, Any]
    rows: list[dict[str, Any]]
    json_path: Path | None = None
    csv_path: Path | None = None

    @property
    def evaluation_id(self) -> str:
        return str(self.metadata["evaluation_id"])

    def to_artifact(self) -> dict[str, Any]:
        return {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "artifact_type": EVALUATION_ARTIFACT_TYPE,
            "metadata": self.metadata,
            "metrics": self.metrics,
            "rows": self.rows,
        }


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _artifact_id(timestamp: datetime, suffix: str) -> str:
    return f"{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}_{suffix}"


def _sha256_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _image_sha256(path: Path, recorded: str | None) -> str | None:
    if recorded:
        return recorded
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prediction_roots(cfg: Config) -> tuple[Path, Path]:
    results = cfg.root / "data" / cfg.dataset_id / "results"
    return results / "predictions", results / "evaluations"


def _active_grippers(batch: BenchmarkPredictionBatch) -> tuple[Gripper, ...]:
    return tuple(Gripper(name) for name in batch.metadata["active_grippers"])


def _batch_config(cfg: Config, batch: BenchmarkPredictionBatch) -> Config:
    if batch.metadata["dataset_id"] != cfg.dataset_id:
        raise ValueError(
            f"prediction batch belongs to {batch.metadata['dataset_id']!r}, "
            f"not {cfg.dataset_id!r}"
        )
    scoped = cfg.model_copy(deep=True)
    scoped.prediction.active_grippers = _active_grippers(batch)
    return scoped


def benchmark_scope(cfg: Config, experiment: str) -> BenchmarkScope:
    """Resolve the CSV-defined holdout, with legacy leave-one-out fallback."""
    experiment_id = experiment.lower()
    dataset = get_dataset(cfg, cfg.dataset_id)
    eligibility = experiment_eligibility(dataset, cfg, experiment_id)
    train_ids = tuple(
        sorted(item.object_id for item in dataset.objects.values() if item.split == "train")
    )
    test_ids = tuple(
        sorted(item.object_id for item in dataset.objects.values() if item.split == "test")
    )
    if not test_ids:
        return BenchmarkScope(
            mode="leave_one_surface_out",
            train_ids=train_ids,
            test_ids=(),
            query_ids=eligibility.query_ids,
            reference_ids=eligibility.reference_ids,
            skipped_queries=eligibility.skipped_queries,
        )

    query_ready = set(eligibility.query_ids)
    queries = [object_id for object_id in test_ids if object_id in query_ready]
    references = tuple(
        object_id
        for object_id in eligibility.reference_ids
        if dataset.objects[object_id].split == "train"
    )
    skipped = {
        object_id: eligibility.skipped_queries[object_id]
        for object_id in test_ids
        if object_id in eligibility.skipped_queries
    }
    if experiment_id in {"e3", "e4"} and not references:
        reason = ("no eligible training reference object",)
        for object_id in queries:
            skipped[object_id] = reason
        queries = []
    return BenchmarkScope(
        mode="fixed_train_test_holdout",
        train_ids=train_ids,
        test_ids=test_ids,
        query_ids=tuple(queries),
        reference_ids=references,
        skipped_queries=skipped,
    )


def generate_benchmark_predictions(
    cfg: Config,
    experiment: str,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> BenchmarkPredictionBatch:
    """Generate immutable predictions for held-out, query-ready objects."""
    experiment_id = experiment.lower()
    dataset = get_dataset(cfg, cfg.dataset_id)
    scope = benchmark_scope(cfg, experiment_id)
    object_ids = list(scope.query_ids)
    if not object_ids:
        raise ValueError(f"no held-out query-ready objects for {experiment_id.upper()}")

    records = load_experience_pool(cfg)
    reference_ids = set(scope.reference_ids)
    train_ids = set(scope.train_ids)
    fixed_holdout = scope.mode == "fixed_train_test_holdout"
    active_grippers = cfg.prediction.active_grippers
    joint_mode = len(active_grippers) == 2
    created_at = _utc_now()
    batch_id = _artifact_id(created_at, experiment_id)
    rows: list[dict[str, Any]] = []
    input_snapshot: list[dict[str, Any]] = []

    for index, object_id in enumerate(object_ids, start=1):
        item = dataset.objects[object_id]
        train = [
            record
            for record in records
            if (
                record.object_id in train_ids
                if fixed_holdout
                else record.surface_id != item.surface_id
            )
            and (
                experiment_id not in {"e3", "e4"}
                or record.object_id in reference_ids
            )
        ]
        image_path = cfg.root / item.image.path
        import cv2

        image = cv2.imread(str(image_path))
        image_digest = _image_sha256(image_path, item.image.sha256)
        semantic_description = (
            item.description.value.description if item.description is not None else None
        )
        query_snapshot = {
            "object_id": object_id,
            "surface_id": item.surface_id,
            "condition_id": item.condition_id,
            "object_name": item.name,
            "image_path": item.image.path,
            "image_sha256": image_digest,
            "mass_g": item.mass_g,
            "roughness_index": item.roughness_index,
            "projected_contact_fraction": item.projected_contact_fraction,
            "semantic_description": semantic_description,
        }
        detailed = Pipeline(cfg, experiment_id).fit(train).predict_detailed(
            QueryInput(
                object_id=object_id,
                surface_id=item.surface_id,
                condition_id=item.condition_id,
                mass_g=item.mass_g,
                roughness_index=item.roughness_index,
                projected_contact_fraction=item.projected_contact_fraction,
                image_bgr=image,
                image_path=item.image.path,
                semantic_description=semantic_description,
            )
        )
        result = detailed.selection
        row: dict[str, Any] = {
            **query_snapshot,
            "active_grippers": [gripper.value for gripper in active_grippers],
            "generation_mode": detailed.generation_mode,
            "predicted_gripper": result.desired_gripper,
            "predicted_normal_force_n": result.predicted_normal_force_n,
            "model_recommended_gripper": result.model_recommended_gripper,
            "recommendation_agrees_with_selector": (
                result.recommendation_agrees_with_selector
            ),
            "pipeline_result": pipeline_result_to_dict(detailed),
        }
        for gripper in active_grippers:
            prediction = result.candidate_predictions[gripper.value]
            row.update(
                {
                    f"pred_{gripper.value}_force_n": (
                        prediction.predicted_normal_force_n
                    ),
                    f"pred_{gripper.value}_feasible": prediction.feasible,
                }
            )
        rows.append(row)
        input_snapshot.append(
            {
                **query_snapshot,
                "retrieved_objects": row["pipeline_result"]["retrieved_objects"],
            }
        )
        if progress is not None:
            progress(index, len(object_ids), object_id)

    definition = cfg.experiment(experiment_id)
    retrieval_mode = EXPERIMENT_CATALOG[experiment_id].retrieval_mode
    prompt_context = prompt_provenance(cfg, definition.prompt)
    metadata = {
        "batch_id": batch_id,
        "dataset_id": cfg.dataset_id,
        "created_at": created_at.isoformat(),
        "experiment": experiment_id,
        "experiment_method": definition.method.value,
        "experiment_definition_version": EXPERIMENT_DEFINITION_VERSION,
        "experiment_definition": definition.model_dump(mode="json"),
        "backend": backend_provenance(cfg, experiment_id),
        "generation_source_sha256": source_sha256(cfg),
        "generation_input_sha256": _sha256_payload(
            {
                "experiment": experiment_id,
                "evaluation_protocol": scope.mode,
                "active_grippers": [gripper.value for gripper in active_grippers],
                "queries": input_snapshot,
                "reference_ids": sorted(reference_ids),
                "train_ids": scope.train_ids,
                "test_ids": scope.test_ids,
            }
        ),
        "evaluation_protocol": scope.mode,
        "prediction_protocol": (
            "CSV-defined fixed train/test holdout"
            if fixed_holdout
            else "query-excluded reference generation"
        ),
        "split_sha256": _sha256_payload(
            {"train": scope.train_ids, "test": scope.test_ids}
        ),
        "train_ids": list(scope.train_ids),
        "test_ids": list(scope.test_ids),
        "query_count": len(rows),
        "eligible_query_ids": object_ids,
        "reference_ids": sorted(reference_ids),
        "skipped_queries": scope.skipped_queries,
        "model": cfg.models.vlm,
        "embedding_model": cfg.retrieval.embedding.model,
        "embedding_dim": cfg.retrieval.embedding.dim,
        "inputs": cfg.inputs.model_dump(mode="json"),
        "roughness_measurement": cfg.roughness.model_dump(mode="json"),
        "active_grippers": [gripper.value for gripper in active_grippers],
        "generation_mode": "joint" if joint_mode else "single",
        "retrieval": cfg.retrieval.model_dump(mode="json"),
        "retrieval_mode": retrieval_mode.value if retrieval_mode is not None else None,
        "prediction_prompts": prompt_context["prediction"],
        "prompt_context": prompt_context,
    }
    return BenchmarkPredictionBatch(metadata=metadata, rows=rows)


def _truth_snapshot(
    cfg: Config,
    batch: BenchmarkPredictionBatch,
) -> tuple[dict[str, Any], str]:
    scoped = _batch_config(cfg, batch)
    dataset = get_dataset(scoped, scoped.dataset_id)
    payload: dict[str, Any] = {}
    for row in batch.rows:
        object_id = row["object_id"]
        item = dataset.objects.get(object_id)
        payload[object_id] = {
            gripper.value: (
                item.gripper_outcomes[gripper].model_dump(mode="json")
                if item is not None and gripper in item.gripper_outcomes
                else None
            )
            for gripper in scoped.prediction.active_grippers
        }
    return payload, _sha256_payload(payload)


def truth_snapshot_sha256(cfg: Config, batch: BenchmarkPredictionBatch) -> str:
    """Hash only truth for predicted objects and active targets."""
    return _truth_snapshot(cfg, batch)[1]


def evaluation_readiness(
    cfg: Config,
    batch: BenchmarkPredictionBatch,
):  # noqa: ANN201
    scoped = _batch_config(cfg, batch)
    dataset = get_dataset(scoped, scoped.dataset_id)
    return evaluation_truth_eligibility(
        dataset,
        scoped,
        [row["object_id"] for row in batch.rows],
    )


def _evaluation_row(
    source: dict[str, Any],
    truth,  # noqa: ANN001
    result: SelectionResult,
    active_grippers: tuple[Gripper, ...],
) -> dict[str, Any]:
    joint_mode = len(active_grippers) == 2
    optimal: set[Gripper] = set()
    oracle_force: float | None = None
    if joint_mode:
        optimal, oracle_force = truth.optimal_grippers()
    chosen = result.desired_gripper
    chosen_truth = (
        truth.get(Gripper(chosen)) if chosen in {"gecko", "silicone"} else None
    )
    regret = None
    if joint_mode and oracle_force is not None:
        regret = (
            (chosen_truth.min_force_n or 0.0) - oracle_force
            if chosen_truth is not None and chosen_truth.feasible
            else None
        )
    row = {
        **source,
        "true_favored": next(iter(optimal)).value if joint_mode else None,
        "selection_correct": (
            chosen in {gripper.value for gripper in optimal} if joint_mode else None
        ),
        "regret_n": regret,
    }
    for gripper in active_grippers:
        truth_record = truth.get(gripper)
        true_force = truth_record.min_force_n if truth_record else None
        predicted_force = float(source[f"pred_{gripper.value}_force_n"])
        row.update(
            {
                f"true_{gripper.value}_force_n": true_force,
                f"true_{gripper.value}_feasible": (
                    truth_record.feasible if truth_record else None
                ),
                f"error_{gripper.value}_n": (
                    predicted_force - true_force if true_force is not None else None
                ),
                f"absolute_error_{gripper.value}_n": (
                    abs(predicted_force - true_force) if true_force is not None else None
                ),
            }
        )
    return row


def evaluate_benchmark_predictions(
    cfg: Config,
    batch: BenchmarkPredictionBatch,
) -> BenchmarkEvaluation:
    """Score a saved prediction batch against currently available truth."""
    scoped = _batch_config(cfg, batch)
    dataset = get_dataset(scoped, scoped.dataset_id)
    readiness = evaluation_truth_eligibility(
        dataset,
        scoped,
        [row["object_id"] for row in batch.rows],
    )
    if not readiness.eligible_ids:
        raise ValueError("no predicted objects currently have evaluation-ready truth")

    truth_records = dataset_experience_records(dataset, scoped.force.limit_n)
    truth_by_object = group_by_object(truth_records)
    eligible = set(readiness.eligible_ids)
    eval_rows: list[EvalRow] = []
    output_rows: list[dict[str, Any]] = []
    active_grippers = scoped.prediction.active_grippers
    for source in batch.rows:
        object_id = source["object_id"]
        if object_id not in eligible:
            continue
        truth = truth_by_object[object_id]
        result = pipeline_result_from_dict(source["pipeline_result"]).selection
        eval_rows.append(EvalRow(object_id=object_id, truth=truth, result=result))
        output_rows.append(_evaluation_row(source, truth, result, active_grippers))

    _, truth_hash = _truth_snapshot(scoped, batch)
    created_at = _utc_now()
    evaluation_id = _artifact_id(created_at, truth_hash[:12])
    metadata = {
        "evaluation_id": evaluation_id,
        "prediction_batch_id": batch.batch_id,
        "dataset_id": scoped.dataset_id,
        "created_at": created_at.isoformat(),
        "experiment": batch.metadata["experiment"],
        "experiment_method": batch.metadata["experiment_method"],
        "experiment_definition_version": batch.metadata[
            "experiment_definition_version"
        ],
        "active_grippers": [gripper.value for gripper in active_grippers],
        "generation_mode": batch.metadata["generation_mode"],
        "evaluation_protocol": batch.metadata.get(
            "evaluation_protocol", "leave_one_surface_out"
        ),
        "backend": batch.metadata["backend"],
        "prediction_created_at": batch.metadata["created_at"],
        "generation_source_sha256": batch.metadata["generation_source_sha256"],
        "generation_input_sha256": batch.metadata["generation_input_sha256"],
        "truth_source_sha256": dataset.source_fingerprint,
        "truth_snapshot_sha256": truth_hash,
        "coverage": {
            "predicted": len(batch.rows),
            "evaluated": len(output_rows),
            "skipped": len(readiness.skipped),
            "fraction": len(output_rows) / len(batch.rows),
        },
        "skipped_truth": readiness.skipped,
        "exports": {},
    }
    return BenchmarkEvaluation(
        metadata=metadata,
        metrics=compute_metrics(eval_rows, scoped).to_dict(),
        rows=output_rows,
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    excluded = {"pipeline_result"}
    fields = [key for key in rows[0] if key not in excluded]
    with tempfile.NamedTemporaryFile(
        "w",
        newline="",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            [{key: _csv_value(value) for key, value in row.items()} for row in rows]
        )
        temporary = Path(fh.name)
    temporary.replace(path)


def save_prediction_batch(
    cfg: Config,
    batch: BenchmarkPredictionBatch,
) -> tuple[Path, Path]:
    predictions_root, _ = _prediction_roots(cfg)
    json_path = predictions_root / f"{batch.batch_id}.json"
    csv_path = predictions_root / f"{batch.batch_id}.csv"
    if json_path.exists() or csv_path.exists():
        raise FileExistsError(f"prediction batch {batch.batch_id!r} already exists")
    _write_rows_csv(csv_path, batch.rows)
    write_json_atomic(json_path, batch.to_artifact())
    batch.json_path = json_path
    batch.csv_path = csv_path
    return json_path, csv_path


def load_prediction_batch(path: str | Path) -> BenchmarkPredictionBatch:
    json_path = Path(path)
    artifact = json.loads(json_path.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise ValueError(
            f"only schema-v{BENCHMARK_SCHEMA_VERSION} prediction batches are loadable"
        )
    if artifact.get("artifact_type") != PREDICTION_ARTIFACT_TYPE:
        raise ValueError("artifact is not a benchmark prediction batch")
    batch = BenchmarkPredictionBatch(
        metadata=artifact["metadata"],
        rows=artifact["rows"],
        json_path=json_path,
        csv_path=json_path.with_suffix(".csv"),
    )
    return batch


def list_prediction_batches(
    cfg: Config,
    *,
    experiment: str | None = None,
) -> list[BenchmarkPredictionBatch]:
    predictions_root, _ = _prediction_roots(cfg)
    batches: list[BenchmarkPredictionBatch] = []
    for path in sorted(predictions_root.glob("*.json"), reverse=True):
        try:
            batch = load_prediction_batch(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if experiment is None or batch.metadata["experiment"] == experiment.lower():
            batches.append(batch)
    return batches


def save_benchmark_evaluation(
    cfg: Config,
    evaluation: BenchmarkEvaluation,
) -> dict[str, Path]:
    _, evaluations_root = _prediction_roots(cfg)
    destination = evaluations_root / evaluation.metadata["prediction_batch_id"]
    destination.mkdir(parents=True, exist_ok=True)
    stem = evaluation.evaluation_id
    json_path = destination / f"{stem}.json"
    csv_path = destination / f"{stem}.csv"
    if json_path.exists() or csv_path.exists():
        raise FileExistsError(f"evaluation {stem!r} already exists")
    _write_rows_csv(csv_path, evaluation.rows)

    from .reporting import export_individual_evaluation

    exports = export_individual_evaluation(evaluation.to_artifact(), destination, stem)
    paths = {"json": json_path, "csv": csv_path, **exports}
    evaluation.metadata["exports"] = {
        name: str(path.relative_to(cfg.root)) for name, path in paths.items()
    }
    write_json_atomic(json_path, evaluation.to_artifact())
    evaluation.json_path = json_path
    evaluation.csv_path = csv_path
    return paths


def load_benchmark_evaluation(path: str | Path) -> BenchmarkEvaluation:
    json_path = Path(path)
    artifact = json.loads(json_path.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise ValueError(
            f"only schema-v{BENCHMARK_SCHEMA_VERSION} evaluations are loadable"
        )
    if artifact.get("artifact_type") != EVALUATION_ARTIFACT_TYPE:
        raise ValueError("artifact is not a benchmark evaluation")
    return BenchmarkEvaluation(
        metadata=artifact["metadata"],
        metrics=artifact["metrics"],
        rows=artifact["rows"],
        json_path=json_path,
        csv_path=json_path.with_suffix(".csv"),
    )


def list_batch_evaluations(
    cfg: Config,
    batch_id: str,
) -> list[BenchmarkEvaluation]:
    _, evaluations_root = _prediction_roots(cfg)
    evaluations: list[BenchmarkEvaluation] = []
    for path in sorted((evaluations_root / batch_id).glob("*.json"), reverse=True):
        try:
            evaluations.append(load_benchmark_evaluation(path))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return evaluations


def get_or_create_benchmark_evaluation(
    cfg: Config,
    batch: BenchmarkPredictionBatch,
) -> tuple[BenchmarkEvaluation, dict[str, Path], bool]:
    """Return an evaluation for current truth, reusing an identical snapshot."""
    truth_hash = truth_snapshot_sha256(cfg, batch)
    for existing in list_batch_evaluations(cfg, batch.batch_id):
        if existing.metadata.get("truth_snapshot_sha256") == truth_hash:
            exports = existing.metadata.get("exports", {})
            paths = {
                name: cfg.root / relative for name, relative in exports.items()
            }
            return existing, paths, True
    evaluation = evaluate_benchmark_predictions(cfg, batch)
    paths = save_benchmark_evaluation(cfg, evaluation)
    return evaluation, paths, False
