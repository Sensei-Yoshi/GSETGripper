"""Regression tests for the rich saved-benchmark object inspector."""

from __future__ import annotations

import hashlib

from PIL import Image
from streamlit.testing.v1 import AppTest

from modules.artifacts import pipeline_result_to_dict
from modules.benchmarking import BenchmarkEvaluation, BenchmarkPredictionBatch
from modules.config import load_config
from modules.contracts import Gripper, PerGripperPrediction, SelectionResult
from modules.datasets import get_dataset
from modules.experiments import PipelineRunResult
from streamlit_app.benchmark_inspector import (
    benchmark_image_state,
    benchmark_object_options,
    build_benchmark_inspection,
)
from streamlit_app.context import AppContext


def _render_inspector(cfg, batch, evaluation):  # noqa: ANN001, ANN201
    from streamlit_app.benchmark_inspector import render_benchmark_object_inspector

    render_benchmark_object_inspector(cfg, batch, evaluation)


def _render_batch(context, batch):  # noqa: ANN001, ANN201
    from streamlit_app.tabs.runs_viewer import _render_prediction_batch

    _render_prediction_batch(context, batch)


def _write_test_image(path) -> str:  # noqa: ANN001
    Image.new("RGB", (4, 4), color=(12, 34, 56)).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _batch(cfg, image_path, image_sha256: str) -> BenchmarkPredictionBatch:  # noqa: ANN001
    prediction = PerGripperPrediction(
        candidate_gripper=Gripper.GECKO,
        predicted_normal_force_n=2.25,
        reasoning_trace="Saved benchmark reasoning",
    )
    selection = SelectionResult(
        desired_gripper="gecko",
        predicted_normal_force_n=2.25,
        candidate_predictions={"gecko": prediction},
    )
    detailed = PipelineRunResult(
        experiment_id="e3",
        experiment_method="hybrid_retrieval_vlm",
        experiment_definition_version=8,
        selection=selection,
        semantic_description="saved object description",
        retrieved_objects=[],
        cache_stats={},
        active_grippers=("gecko",),
        generation_mode="single",
        retrieval_mode="hybrid",
        effective_inputs=("mass", "roughness"),
        object_id="held_out_box",
        surface_id="held_out_box",
        condition_id="baseline",
    )
    row = {
        "object_id": "held_out_box",
        "surface_id": "held_out_box",
        "condition_id": "baseline",
        "object_name": "Held Out Box",
        "image_path": str(image_path.relative_to(cfg.root)),
        "image_sha256": image_sha256,
        "mass_g": 231.63,
        "roughness_index": 1544.73,
        "projected_contact_fraction": None,
        "semantic_description": "saved object description",
        "active_grippers": ["gecko"],
        "generation_mode": "single",
        "predicted_gripper": "gecko",
        "predicted_normal_force_n": 2.25,
        "model_recommended_gripper": None,
        "recommendation_agrees_with_selector": None,
        "pred_gecko_force_n": 2.25,
        "pred_gecko_feasible": True,
        "pipeline_result": pipeline_result_to_dict(detailed),
    }
    metadata = {
        "batch_id": "saved_e3_batch",
        "dataset_id": cfg.dataset_id,
        "created_at": "2026-07-29T12:00:00+00:00",
        "experiment": "e3",
        "experiment_method": "hybrid_retrieval_vlm",
        "experiment_definition": {"method": "hybrid_retrieval_vlm", "prompt": "e3"},
        "evaluation_protocol": "fixed_train_test_holdout",
        "active_grippers": ["gecko"],
        "backend": {"force": "test"},
        "model": "saved-model",
        "embedding_model": "saved-embedding",
        "inputs": {
            "use_roughness": False,
            "use_projected_contact": False,
        },
        "retrieval": {
            **cfg.retrieval.model_dump(mode="json"),
            "k": 2,
        },
        "reference_ids": ["training_object"],
        "test_ids": ["held_out_box"],
    }
    return BenchmarkPredictionBatch(metadata=metadata, rows=[row])


def _evaluation(batch: BenchmarkPredictionBatch) -> BenchmarkEvaluation:
    row = {
        **batch.rows[0],
        "true_favored": None,
        "selection_correct": None,
        "regret_n": None,
        "true_gecko_force_n": 2.0,
        "true_gecko_feasible": True,
        "error_gecko_n": 0.25,
        "absolute_error_gecko_n": 0.25,
    }
    return BenchmarkEvaluation(
        metadata={
            "evaluation_id": "saved_truth_version",
            "coverage": {"predicted": 1, "evaluated": 1},
        },
        metrics={},
        rows=[row],
    )


def _paired_batch_and_evaluation(
    cfg,
    image_path,
    image_sha256: str,
) -> tuple[BenchmarkPredictionBatch, BenchmarkEvaluation]:  # noqa: ANN001
    batch = _batch(cfg, image_path, image_sha256)
    predictions = {
        "gecko": PerGripperPrediction(
            candidate_gripper=Gripper.GECKO,
            predicted_normal_force_n=3.5,
            reasoning_trace="Gecko evidence",
        ),
        "silicone": PerGripperPrediction(
            candidate_gripper=Gripper.SILICONE,
            predicted_normal_force_n=2.5,
            reasoning_trace="Silicone evidence",
        ),
    }
    detailed = PipelineRunResult(
        experiment_id="e3",
        experiment_method="hybrid_retrieval_vlm",
        experiment_definition_version=8,
        selection=SelectionResult(
            desired_gripper="silicone",
            predicted_normal_force_n=2.5,
            candidate_predictions=predictions,
        ),
        semantic_description="saved object description",
        retrieved_objects=[],
        cache_stats={},
        active_grippers=("gecko", "silicone"),
        generation_mode="joint",
        retrieval_mode="hybrid",
        object_id="held_out_box",
        surface_id="held_out_box",
    )
    batch.metadata["active_grippers"] = ["gecko", "silicone"]
    batch.rows[0].update(
        {
            "active_grippers": ["gecko", "silicone"],
            "generation_mode": "joint",
            "predicted_gripper": "silicone",
            "predicted_normal_force_n": 2.5,
            "pred_gecko_force_n": 3.5,
            "pred_gecko_feasible": True,
            "pred_silicone_force_n": 2.5,
            "pred_silicone_feasible": True,
            "pipeline_result": pipeline_result_to_dict(detailed),
        }
    )
    evaluation_row = {
        **batch.rows[0],
        "true_favored": "silicone",
        "selection_correct": True,
        "regret_n": 0.0,
        "true_gecko_force_n": None,
        "true_gecko_feasible": False,
        "error_gecko_n": None,
        "absolute_error_gecko_n": None,
        "true_silicone_force_n": 2.0,
        "true_silicone_feasible": True,
        "error_silicone_n": 0.5,
        "absolute_error_silicone_n": 0.5,
    }
    evaluation = BenchmarkEvaluation(
        metadata={
            "evaluation_id": "paired_truth_version",
            "coverage": {"predicted": 1, "evaluated": 1},
        },
        metrics={},
        rows=[evaluation_row],
    )
    return batch, evaluation


def test_benchmark_inspection_reconstructs_saved_config_truth_and_prediction(tmp_path) -> None:
    cfg = load_config().model_copy(deep=True)
    cfg.root = tmp_path
    cfg.dataset_id = "test_dataset"
    image_path = tmp_path / "data/test_dataset/held_out_box.png"
    image_path.parent.mkdir(parents=True)
    digest = _write_test_image(image_path)
    batch = _batch(cfg, image_path, digest)

    inspection = build_benchmark_inspection(cfg, batch, batch.rows[0], _evaluation(batch))

    assert inspection.config is not cfg
    assert inspection.config.retrieval.k == 2
    assert inspection.config.inputs.use_roughness is False
    assert inspection.config.prediction.active_grippers == (Gripper.GECKO,)
    assert inspection.truth is not None
    assert inspection.truth.gecko is not None
    assert inspection.truth.gecko.min_force_n == 2.0
    assert inspection.detailed.selection.predicted_normal_force_n == 2.25
    assert inspection.image.status == "verified"
    assert list(benchmark_object_options(batch.rows)) == [
        "Held Out Box · held_out_box"
    ]


def test_benchmark_inspection_reconstructs_paired_and_infeasible_truth(tmp_path) -> None:
    cfg = load_config().model_copy(deep=True)
    cfg.root = tmp_path
    cfg.dataset_id = "test_dataset"
    image_path = tmp_path / "data/test_dataset/held_out_box.png"
    image_path.parent.mkdir(parents=True)
    batch, evaluation = _paired_batch_and_evaluation(
        cfg,
        image_path,
        _write_test_image(image_path),
    )

    inspection = build_benchmark_inspection(cfg, batch, batch.rows[0], evaluation)

    assert inspection.config.prediction.active_grippers == (
        Gripper.GECKO,
        Gripper.SILICONE,
    )
    assert inspection.detailed.active_grippers == ("gecko", "silicone")
    assert inspection.truth is not None
    assert inspection.truth.gecko is not None
    assert inspection.truth.gecko.feasible is False
    assert inspection.truth.gecko.failed_at_limit_n == cfg.force.limit_n
    assert inspection.truth.silicone is not None
    assert inspection.truth.silicone.min_force_n == 2.0
    assert inspection.truth.optimal_grippers()[0] == {Gripper.SILICONE}


def test_benchmark_image_integrity_reports_changed_and_missing_files(tmp_path) -> None:
    cfg = load_config().model_copy(deep=True)
    cfg.root = tmp_path
    cfg.dataset_id = "test_dataset"
    image_path = tmp_path / "data/test_dataset/held_out_box.png"
    image_path.parent.mkdir(parents=True)
    batch = _batch(cfg, image_path, _write_test_image(image_path))

    image_path.write_bytes(b"changed-image")
    assert benchmark_image_state(cfg, batch.rows[0]).status == "changed"
    image_path.unlink()
    assert benchmark_image_state(cfg, batch.rows[0]).status == "missing"


def test_benchmark_object_inspector_renders_evaluated_and_unevaluated_states(
    tmp_path,
) -> None:
    cfg = load_config().model_copy(deep=True)
    cfg.root = tmp_path
    cfg.dataset_id = "test_dataset"
    image_path = tmp_path / "data/test_dataset/held_out_box.png"
    image_path.parent.mkdir(parents=True)
    batch = _batch(cfg, image_path, _write_test_image(image_path))

    evaluated = AppTest.from_function(
        _render_inspector,
        args=(cfg, batch, _evaluation(batch)),
    ).run(timeout=10)
    assert list(evaluated.exception) == []
    assert [item.label for item in evaluated.selectbox] == ["Benchmark object"]
    assert any(item.label == "Predicted force" for item in evaluated.metric)
    assert any(item.label == "True gecko force" for item in evaluated.metric)
    assert any("saved_truth_version" in item.value for item in evaluated.caption)
    assert any(
        "Selected benchmark evaluation truth" in item.value
        for item in evaluated.markdown
    )

    unevaluated = AppTest.from_function(
        _render_inspector,
        args=(cfg, batch, None),
    ).run(timeout=10)
    assert list(unevaluated.exception) == []
    assert any("no saved evaluation version" in item.value for item in unevaluated.markdown)


def test_prediction_batch_view_contains_object_inspector_subtab(tmp_path) -> None:
    cfg = load_config().model_copy(deep=True)
    cfg.root = tmp_path
    cfg.dataset_id = "test_dataset"
    image_path = tmp_path / "data/test_dataset/held_out_box.png"
    image_path.parent.mkdir(parents=True)
    batch = _batch(cfg, image_path, _write_test_image(image_path))
    dataset = get_dataset(cfg, cfg.dataset_id)
    context = AppContext(
        config=dataset.runtime_config(cfg),
        catalog=(dataset,),
        dataset=dataset,
        rows=list(dataset.objects.values()),
        summary=dataset.summary(),
    )

    app = AppTest.from_function(_render_batch, args=(context, batch)).run(timeout=10)

    assert list(app.exception) == []
    assert [tab.label for tab in app.tabs] == [
        "Summary",
        "Object Inspector",
        "Plots",
        "Evaluated Rows",
        "All Predictions",
        "Provenance",
    ]
    assert any(item.label == "Benchmark object" for item in app.selectbox)
