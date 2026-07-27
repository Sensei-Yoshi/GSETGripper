from __future__ import annotations

import shutil

import pytest

from modules.config import (
    PromptBundle,
    load_config,
    load_prompt_bundle,
    save_prompt_bundle,
)
from modules.expforce import source_path
from modules.reporting import (
    calibration_figure,
    common_intersection_artifacts,
    comparison_rows,
    export_comparison,
    metrics_rows,
)
from modules.suites import create_suite, load_suite, run_suite, suite_manifest_path


def _artifact(offset: float) -> dict:
    force = {
        "gecko": {"n": 2, "mae": offset, "rmse": offset, "medae": offset},
        "silicone": {"n": 2, "mae": offset, "rmse": offset, "medae": offset},
        "overall": {"n": 4, "mae": offset, "rmse": offset, "medae": offset},
    }
    return {
        "metrics": {
            "force": force,
            "selection": {
                "accuracy": 1.0,
                "infeasible_pick_rate": 0.0,
                "mean_regret_n": 0.0,
                "median_regret_n": 0.0,
                "worst_regret_n": 0.0,
            },
            "model_recommendation": {"accuracy": 1.0, "selector_agreement": 1.0},
        },
        "rows": [
            {
                "object_id": "object_a",
                "true_gecko_force_n": 1.0,
                "true_gecko_feasible": True,
                "pred_gecko_force_n": 1.0 + offset,
                "pred_gecko_feasible": True,
                "true_silicone_force_n": 1.5,
                "true_silicone_feasible": True,
                "pred_silicone_force_n": 1.5 + offset,
                "pred_silicone_feasible": True,
                "true_favored": "gecko",
                "predicted_gripper": "gecko",
                "selection_correct": True,
                "regret_n": 0.0,
            },
            {
                "object_id": "object_b",
                "true_gecko_force_n": 2.0,
                "true_gecko_feasible": True,
                "pred_gecko_force_n": 2.0 + offset,
                "pred_gecko_feasible": True,
                "true_silicone_force_n": 2.5,
                "true_silicone_feasible": True,
                "pred_silicone_force_n": 2.5 + offset,
                "pred_silicone_feasible": True,
                "true_favored": "gecko",
                "predicted_gripper": "gecko",
                "selection_correct": True,
                "regret_n": 0.0,
            },
        ],
    }


def test_prompt_bundle_round_trip(tmp_path):
    source = load_prompt_bundle()
    destination = tmp_path / "prompts.yaml"
    save_prompt_bundle(PromptBundle.model_validate(source), destination)
    loaded = load_prompt_bundle(destination)

    assert loaded == source
    assert set(loaded.prompts.experiments) == {"e1", "e2", "e3", "e4"}
    assert set(loaded.embodiments) == {"gecko", "silicone"}


def test_suite_manifest_snapshots_primary_experiments(tmp_path):
    source_cfg = load_config().model_copy(deep=True)
    cfg = source_cfg.model_copy(deep=True)
    cfg.root = tmp_path
    dataset = tmp_path / "data/expforce/dataset.csv"
    dataset.parent.mkdir(parents=True)
    shutil.copyfile(source_path(source_cfg), dataset)

    manifest = create_suite(cfg)
    persisted = load_suite(suite_manifest_path(cfg, manifest["suite_id"]))

    assert persisted["experiments"] == ["e1", "e2", "e3", "e4"]
    assert persisted["snapshot"]["experiment_definition_version"] == 6
    assert persisted["schema_version"] == 7
    assert persisted["backend"] == "gemini_joint_generation"
    assert persisted["snapshot"]["backend"] == "gemini_joint_generation"
    assert persisted["snapshot"]["prompt_bundle_sha256"]
    assert set(persisted["prompt_context"]["experiment_instructions"]) == {
        "e1",
        "e2",
        "e3",
        "e4",
    }
    assert all(state["status"] == "pending" for state in persisted["runs"].values())
    assert set(persisted["snapshot"]["eligibility"]) == {"e1", "e2", "e3", "e4"}


def test_legacy_suite_is_read_only():
    with pytest.raises(ValueError, match="Legacy suites are read-only"):
        run_suite(load_config(), {"schema_version": 5})


def test_suite_with_no_eligible_rows_marks_each_experiment_skipped(tmp_path) -> None:
    source_cfg = load_config().model_copy(deep=True)
    cfg = source_cfg.model_copy(deep=True)
    cfg.root = tmp_path
    dataset = tmp_path / "data/expforce/dataset.csv"
    dataset.parent.mkdir(parents=True)
    shutil.copyfile(source_path(source_cfg), dataset)

    completed = run_suite(cfg, create_suite(cfg))

    assert completed["status"] == "completed"
    assert {state["status"] for state in completed["runs"].values()} == {"skipped"}


def test_common_intersection_recomputes_metrics_on_shared_objects() -> None:
    artifacts = {
        experiment: _artifact(index / 10)
        for index, experiment in enumerate(("e1", "e2", "e3", "e4"), start=1)
    }
    artifacts["e4"]["rows"] = artifacts["e4"]["rows"][:1]

    comparable, object_ids = common_intersection_artifacts(artifacts, load_config())

    assert object_ids == ("object_a",)
    assert all(len(artifact["rows"]) == 1 for artifact in comparable.values())
    assert all(
        artifact["metrics"]["force"]["overall"]["n"] == 2
        for artifact in comparable.values()
    )


def test_calibration_export_has_eight_panels_and_no_identity_lines(tmp_path):
    artifacts = {
        experiment: _artifact(index / 10)
        for index, experiment in enumerate(("e1", "e2", "e3", "e4"), start=1)
    }
    figure = calibration_figure(artifacts)

    assert len(figure.axes) == 8
    assert all(not axis.lines for axis in figure.axes)
    assert len(comparison_rows(artifacts)) == 16
    assert len(metrics_rows(artifacts)) == 4

    exports = export_comparison(artifacts, tmp_path / "exports")
    assert set(exports) == {"png", "svg", "data_csv", "metrics_csv"}
    assert all((tmp_path / "exports" / path.split("/")[-1]).is_file() for path in exports.values())
