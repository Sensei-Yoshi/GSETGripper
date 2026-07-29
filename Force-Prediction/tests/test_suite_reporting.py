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
    individual_calibration_figure,
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


def _single_artifact(offset: float) -> dict:
    return {
        "metadata": {"active_grippers": ["silicone"], "generation_mode": "single"},
        "metrics": {
            "force": {
                "silicone": {"n": 2, "mae": offset, "rmse": offset, "medae": offset},
                "overall": {"n": 2, "mae": offset, "rmse": offset, "medae": offset},
            },
            "selection": {"applicable": False, "n": 0},
            "model_recommendation": {"applicable": False, "n": 0},
        },
        "rows": [
            {
                "object_id": object_id,
                "active_grippers": ["silicone"],
                "true_silicone_force_n": truth,
                "true_silicone_feasible": True,
                "pred_silicone_force_n": truth + offset,
                "pred_silicone_feasible": True,
                "predicted_gripper": "silicone",
            }
            for object_id, truth in (("object_a", 1.5), ("object_b", 2.5))
        ],
    }


def test_prompt_bundle_round_trip(tmp_path):
    source = load_prompt_bundle()
    destination = tmp_path / "prompts.yaml"
    save_prompt_bundle(PromptBundle.model_validate(source), destination)
    loaded = load_prompt_bundle(destination)

    assert loaded == source
    assert set(loaded.prompts.experiments) == {"e1", "e2", "e3", "e4"}
    assert set(loaded.prompts.target_instructions) == {"single", "joint"}
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
    assert persisted["definition_snapshot"]["experiment_definition_version"] == 9
    assert persisted["schema_version"] == 10
    assert persisted["backend"] == "gemini_joint_generation"
    assert persisted["definition_snapshot"]["backend"] == "gemini_joint_generation"
    assert persisted["active_grippers"] == ["gecko", "silicone"]
    assert persisted["definition_snapshot"]["prompt_bundle_sha256"]
    assert len(persisted["definition_snapshot"]["split"]["train"]) == 129
    assert persisted["definition_snapshot"]["split"]["test"] == []
    assert persisted["definition_snapshot"]["split_sha256"]
    assert set(persisted["prompt_context"]["experiment_instructions"]) == {
        "e1",
        "e2",
        "e3",
        "e4",
    }
    assert all(state["status"] == "pending" for state in persisted["runs"].values())
    assert persisted["evaluations"] == []
    assert "source_sha256" not in persisted["definition_snapshot"]


def test_legacy_suite_is_read_only():
    with pytest.raises(ValueError, match="Legacy suites are read-only"):
        run_suite(load_config(), {"schema_version": 5})


def test_suite_with_no_query_ready_rows_remains_resumable(tmp_path) -> None:
    source_cfg = load_config().model_copy(deep=True)
    cfg = source_cfg.model_copy(deep=True)
    cfg.root = tmp_path
    dataset = tmp_path / "data/expforce/dataset.csv"
    dataset.parent.mkdir(parents=True)
    shutil.copyfile(source_path(source_cfg), dataset)

    completed = run_suite(cfg, create_suite(cfg))

    assert completed["status"] == "waiting"
    assert {state["status"] for state in completed["runs"].values()} == {"waiting"}


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


def test_single_gripper_calibration_has_four_panels():
    artifacts = {
        experiment: _single_artifact(index / 10)
        for index, experiment in enumerate(("e1", "e2", "e3", "e4"), start=1)
    }

    figure = calibration_figure(artifacts)

    assert len(figure.axes) == 4
    assert len(comparison_rows(artifacts)) == 8
    assert all(
        row["active_grippers"] == "silicone" for row in metrics_rows(artifacts)
    )


def test_individual_calibration_uses_one_panel_per_active_gripper():
    paired = _artifact(0.1)
    paired["metadata"] = {"active_grippers": ["gecko", "silicone"], "experiment": "e1"}
    single = _single_artifact(0.1)
    single["metadata"]["experiment"] = "e1"

    assert len(individual_calibration_figure(paired).axes) == 2
    assert len(individual_calibration_figure(single).axes) == 1
