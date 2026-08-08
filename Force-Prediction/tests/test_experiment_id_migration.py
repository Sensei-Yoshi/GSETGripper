from __future__ import annotations

import json

import pytest

from modules.artifacts import (
    PIPELINE_RUN_SCHEMA_VERSION,
    load_saved_runs,
    pipeline_result_from_dict,
)
from modules.benchmarking import BenchmarkPredictionBatch
from modules.config import load_config
from modules.contracts import Gripper, PerGripperPrediction, SelectionResult
from modules.experiment_ids import resolve_experiment_id
from modules.experiments import PipelineRunResult
from modules.suites import (
    SUITE_SCHEMA_VERSION,
    evaluate_suite,
    is_legacy_suite,
    load_suite,
    run_suite_predictions,
    suite_experiments,
)


@pytest.mark.parametrize(
    ("stored", "active"),
    (("e1", "e1"), ("e3", "e2"), ("e4", "e3"), ("e5", "e4"), ("e6", "e5")),
)
def test_definition_v12_ids_resolve_to_contiguous_active_ids(stored, active) -> None:
    assert resolve_experiment_id(stored, 12) == active
    assert resolve_experiment_id(active, 13) == active


def test_versions_other_than_v12_are_not_reinterpreted() -> None:
    assert resolve_experiment_id("e5", 11) == "e5"
    assert resolve_experiment_id("e5", 13) == "e5"
    assert resolve_experiment_id("E5", None) == "e5"


def test_legacy_pipeline_result_resolves_only_its_runtime_id() -> None:
    prediction = PerGripperPrediction(
        candidate_gripper=Gripper.GECKO,
        predicted_normal_force_n=1.5,
    )
    detailed = PipelineRunResult(
        experiment_id="e5",
        experiment_method="hybrid_retrieval_vlm",
        experiment_definition_version=12,
        selection=SelectionResult(
            desired_gripper="gecko",
            predicted_normal_force_n=1.5,
            candidate_predictions={"gecko": prediction},
        ),
        semantic_description="legacy surface",
        retrieved_objects=[],
        cache_stats={},
        active_grippers=("gecko",),
        generation_mode="single",
        retrieval_mode="hybrid",
    )
    payload = {
        "experiment_id": detailed.experiment_id,
        "experiment_method": detailed.experiment_method,
        "experiment_definition_version": detailed.experiment_definition_version,
        "selection": detailed.selection.model_dump(mode="json"),
        "semantic_description": detailed.semantic_description,
        "retrieved_objects": [],
        "cache_stats": {},
        "active_grippers": ["gecko"],
        "generation_mode": "single",
        "retrieval_mode": "hybrid",
    }

    restored = pipeline_result_from_dict(payload)

    assert restored.experiment_id == "e4"
    assert restored.experiment_definition_version == 12


def test_legacy_single_run_is_labeled_without_rewriting_disk(tmp_path) -> None:
    cfg = load_config().model_copy(deep=True)
    cfg.root = tmp_path
    cfg.dataset_id = "Legacy"
    run_path = tmp_path / "data/Legacy/runs/legacy_e5.json"
    run_path.parent.mkdir(parents=True)
    run_path.write_text(
        json.dumps(
            {
                "schema_version": PIPELINE_RUN_SCHEMA_VERSION,
                "experiment": "e5",
                "experiment_definition_version": 12,
                "backend": {"force": "saved"},
            }
        ),
        encoding="utf-8",
    )
    original = run_path.read_bytes()

    runs = load_saved_runs(cfg)

    assert runs[0]["experiment"] == "e5"
    assert runs[0]["active_experiment"] == "e4"
    assert runs[0]["experiment_display_name"].startswith("E4 —")
    assert "legacy v12 ID E5" in runs[0]["experiment_display_name"]
    assert run_path.read_bytes() == original


def test_legacy_benchmark_batch_exposes_active_id_without_mutating_metadata() -> None:
    batch = BenchmarkPredictionBatch(
        metadata={
            "batch_id": "legacy_e6",
            "experiment": "e6",
            "experiment_definition_version": 12,
        },
        rows=[],
    )

    assert batch.experiment_id == "e5"
    assert batch.metadata["experiment"] == "e6"


def test_legacy_suite_loads_with_resolved_keys_but_remains_immutable(tmp_path) -> None:
    stored_ids = ["e1", "e3", "e4", "e5", "e6"]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": SUITE_SCHEMA_VERSION,
                "experiments": stored_ids,
                "definition_snapshot": {
                    "experiment_definition_version": 12,
                    "experiment_definitions": {name: {} for name in stored_ids},
                    "experiment_input_profiles": {name: {} for name in stored_ids},
                },
                "runs": {name: {"status": "completed"} for name in stored_ids},
                "evaluations": [
                    {
                        "artifacts": {name: {} for name in stored_ids},
                        "coverage": {name: {} for name in stored_ids},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    original = manifest_path.read_bytes()

    manifest = load_suite(manifest_path)

    assert is_legacy_suite(manifest)
    assert suite_experiments(manifest) == ("e1", "e2", "e3", "e4", "e5")
    assert set(manifest["runs"]) == {"e1", "e2", "e3", "e4", "e5"}
    assert manifest["legacy_experiment_ids"] == {
        "e1": "e1",
        "e2": "e3",
        "e3": "e4",
        "e4": "e5",
        "e5": "e6",
    }
    assert manifest_path.read_bytes() == original

    cfg = load_config().model_copy(deep=True)
    cfg.root = tmp_path
    with pytest.raises(ValueError, match="immutable legacy"):
        run_suite_predictions(cfg, manifest)
    with pytest.raises(ValueError, match="immutable legacy"):
        evaluate_suite(cfg, manifest)
