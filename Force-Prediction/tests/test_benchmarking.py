from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from modules.benchmarking import (
    benchmark_scope,
    evaluate_benchmark_predictions,
    generate_benchmark_predictions,
    get_or_create_benchmark_evaluation,
    list_batch_evaluations,
    save_prediction_batch,
)
from modules.config import load_config
from modules.contracts import Gripper
from modules.datasets import DatasetObjectEdit, get_dataset, update_dataset_object
from modules.suites import (
    create_suite,
    evaluate_suite,
    run_suite_predictions,
    suite_evaluation_artifacts,
)
from tests.fakes import install_gemini_fakes


def _image_only_config(tmp_path, object_ids=("one", "two")):
    cfg = load_config().model_copy(deep=True)
    cfg.root = tmp_path
    root = tmp_path / "data/ImageOnly/objects"
    ok, encoded = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    for object_id in object_ids:
        image = root / object_id / "image.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(encoded.tobytes())
    dataset = get_dataset(cfg, "ImageOnly")
    return dataset.runtime_config(cfg), dataset


def _complete_edit(index: int, *, gecko_force: float | None = None) -> DatasetObjectEdit:
    return DatasetObjectEdit(
        mass_g=100 + index,
        roughness_index=2 + index,
        projected_contact_fraction=0.65,
        gecko_feasible=True,
        gecko_force_n=gecko_force or 1.0 + index / 10,
        silicone_feasible=True,
        silicone_force_n=1.6 + index / 10,
    )


def test_image_only_e1_generates_then_evaluates_partial_truth_without_model_calls(
    tmp_path,
    monkeypatch,
) -> None:
    cfg, dataset = _image_only_config(tmp_path)
    client = install_gemini_fakes(monkeypatch, cfg.retrieval.embedding.dim)

    batch = generate_benchmark_predictions(
        cfg,
        "e1",
        display_name="  Image-only baseline  ",
    )

    assert len(batch.rows) == 2
    assert batch.display_name == "Image-only baseline"
    assert batch.metadata["display_name"] == "Image-only baseline"
    assert batch.metadata["query_count"] == 2
    assert all(not any(key.startswith("true_") for key in row) for row in batch.rows)
    assert all("pipeline_result" in row for row in batch.rows)
    with pytest.raises(ValueError, match="evaluation-ready truth"):
        evaluate_benchmark_predictions(cfg, batch)

    prediction_paths = save_prediction_batch(cfg, batch)
    prediction_artifact = json.loads(prediction_paths[0].read_text(encoding="utf-8"))
    assert prediction_artifact["schema_version"] == 10
    assert prediction_artifact["artifact_type"] == "benchmark_prediction_batch"
    calls_after_generation = (client.generation_calls, client.embedding_calls)
    dataset = update_dataset_object(cfg, dataset, "one", _complete_edit(1))
    cfg.prediction.active_grippers = (Gripper.GECKO,)

    evaluation, paths, reused = get_or_create_benchmark_evaluation(cfg, batch)

    assert reused is False
    assert evaluation.metadata["coverage"] == {
        "predicted": 2,
        "evaluated": 1,
        "skipped": 1,
        "fraction": 0.5,
    }
    assert len(evaluation.rows) == 1
    assert evaluation.metadata["active_grippers"] == ["gecko", "silicone"]
    assert evaluation.rows[0]["true_gecko_force_n"] == pytest.approx(1.1)
    assert set(paths) == {"json", "csv", "png", "svg"}
    assert all(path.is_file() for path in paths.values())
    evaluation_artifact = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert evaluation_artifact["schema_version"] == 10
    assert evaluation_artifact["artifact_type"] == "benchmark_evaluation"
    assert (client.generation_calls, client.embedding_calls) == calls_after_generation

    same, _, reused = get_or_create_benchmark_evaluation(cfg, batch)
    assert reused is True
    assert same.evaluation_id == evaluation.evaluation_id
    assert len(list_batch_evaluations(cfg, batch.batch_id)) == 1

    dataset = update_dataset_object(
        cfg,
        dataset,
        "one",
        _complete_edit(1, gecko_force=1.25),
    )
    corrected, _, reused = get_or_create_benchmark_evaluation(cfg, batch)
    assert reused is False
    assert corrected.evaluation_id != evaluation.evaluation_id
    assert len(list_batch_evaluations(cfg, batch.batch_id)) == 2
    assert (client.generation_calls, client.embedding_calls) == calls_after_generation


def test_fixed_split_queries_only_test_rows_and_retrieves_only_train_rows(
    tmp_path,
    monkeypatch,
) -> None:
    object_ids = ("train_a", "train_b", "test_a", "test_b")
    cfg, dataset = _image_only_config(tmp_path, object_ids)
    cfg.inputs.roughness_representation = "binary"
    install_gemini_fakes(monkeypatch, cfg.retrieval.embedding.dim)
    for index, object_id in enumerate(object_ids, start=1):
        split = "train" if object_id.startswith("train") else "test"
        dataset = update_dataset_object(
            cfg,
            dataset,
            object_id,
            _complete_edit(index).model_copy(update={"split": split}),
        )

    scope = benchmark_scope(cfg, "e4")
    batch = generate_benchmark_predictions(cfg, "e4")

    assert scope.mode == "fixed_train_test_holdout"
    assert scope.train_ids == ("train_a", "train_b")
    assert scope.test_ids == ("test_a", "test_b")
    assert {row["object_id"] for row in batch.rows} == {"test_a", "test_b"}
    assert batch.metadata["train_ids"] == ["train_a", "train_b"]
    assert batch.metadata["test_ids"] == ["test_a", "test_b"]
    assert batch.metadata["reference_ids"] == ["train_a", "train_b"]
    assert batch.metadata["evaluation_protocol"] == "fixed_train_test_holdout"
    assert batch.metadata["inputs"]["roughness_representation"] == "binary"
    assert batch.metadata["roughness_measurement"]["binary_threshold"] == 1340.0
    assert all(row["roughness_index"] is not None for row in batch.rows)
    assert all(
        {
            retrieved["object_id"]
            for retrieved in row["pipeline_result"]["retrieved_objects"]
        }
        <= {"train_a", "train_b"}
        for row in batch.rows
    )


def test_explicit_blank_benchmark_name_is_rejected(tmp_path) -> None:
    cfg, _ = _image_only_config(tmp_path)

    with pytest.raises(ValueError, match="benchmark name is required"):
        generate_benchmark_predictions(cfg, "e1", display_name="   ")


def test_suite_generates_e1_early_then_resumes_other_experiments(
    tmp_path,
    monkeypatch,
) -> None:
    cfg, dataset = _image_only_config(tmp_path)
    install_gemini_fakes(monkeypatch, cfg.retrieval.embedding.dim)
    manifest = create_suite(cfg)

    manifest = run_suite_predictions(cfg, manifest)

    assert manifest["status"] == "partially_generated"
    assert manifest["runs"]["e1"]["status"] == "completed"
    assert {manifest["runs"][name]["status"] for name in ("e2", "e3", "e4")} == {
        "waiting"
    }
    e1_batch_id = manifest["runs"]["e1"]["prediction_batch_id"]

    for index, object_id in enumerate(("one", "two"), start=1):
        dataset = update_dataset_object(
            cfg,
            dataset,
            object_id,
            _complete_edit(index),
        )

    manifest = run_suite_predictions(cfg, manifest)

    assert manifest["status"] == "predictions_complete"
    assert all(state["status"] == "completed" for state in manifest["runs"].values())
    assert manifest["runs"]["e1"]["prediction_batch_id"] == e1_batch_id

    manifest = evaluate_suite(cfg, manifest)
    artifacts = suite_evaluation_artifacts(cfg, manifest)

    assert manifest["status"] == "evaluated"
    assert set(artifacts) == {"e1", "e2", "e3", "e4"}
    assert manifest["evaluations"][-1]["common_object_ids"] == ["one", "two"]
    assert set(manifest["evaluations"][-1]["exports"]) == {
        "png",
        "svg",
        "data_csv",
        "metrics_csv",
    }
