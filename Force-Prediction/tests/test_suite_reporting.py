from __future__ import annotations

import pytest
from PIL import Image

from modules.config import (
    EXPERIMENT_IDS,
    PromptBundle,
    load_config,
    load_prompt_bundle,
    save_prompt_bundle,
)
from modules.reporting import (
    EXPERIMENT_STYLES,
    calibration_figure,
    common_intersection_artifacts,
    comparison_rows,
    export_comparison,
    force_error_statistics_rows,
    individual_calibration_figure,
    metrics_rows,
    suite_force_by_object_figure,
    suite_percentage_error_figure,
)
from modules.suites import run_suite


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
    assert set(loaded.prompts.experiments) == set(EXPERIMENT_IDS)
    assert set(loaded.prompts.target_instructions) == {"single", "joint"}
    assert set(loaded.embodiments) == {"gecko", "silicone"}


def test_legacy_suite_is_read_only():
    with pytest.raises(ValueError, match="Legacy suites are read-only"):
        run_suite(load_config(), {"schema_version": 5})


def test_common_intersection_recomputes_metrics_on_shared_objects() -> None:
    artifacts = {
        experiment: _artifact(index / 10)
        for index, experiment in enumerate(EXPERIMENT_IDS, start=1)
    }
    artifacts["e6"]["rows"] = artifacts["e6"]["rows"][:1]

    comparable, object_ids = common_intersection_artifacts(artifacts, load_config())

    assert object_ids == ("object_a",)
    assert all(len(artifact["rows"]) == 1 for artifact in comparable.values())
    assert all(
        artifact["metrics"]["force"]["overall"]["n"] == 2
        for artifact in comparable.values()
    )


def test_reporting_handles_experiment_subset(tmp_path):
    subset = ("e1", "e2", "e3", "e4", "e5")
    artifacts = {
        experiment: _artifact(index / 10)
        for index, experiment in enumerate(subset, start=1)
    }

    comparable, object_ids = common_intersection_artifacts(artifacts, load_config())
    assert set(comparable) == set(subset)
    assert object_ids == ("object_a", "object_b")

    # One calibration column per present experiment, times the active grippers.
    assert len(calibration_figure(artifacts).axes) == len(subset) * 2
    assert len(metrics_rows(artifacts)) == len(subset)
    assert suite_force_by_object_figure(artifacts).axes
    assert suite_percentage_error_figure(artifacts).axes

    exports = export_comparison(artifacts, tmp_path / "exports")
    assert all(
        (tmp_path / "exports" / path.split("/")[-1]).is_file()
        for path in exports.values()
    )


def test_calibration_export_has_twelve_panels_and_no_identity_lines(tmp_path):
    artifacts = {
        experiment: _artifact(index / 10)
        for index, experiment in enumerate(EXPERIMENT_IDS, start=1)
    }
    figure = calibration_figure(artifacts)

    assert len(figure.axes) == 12
    assert all(not axis.lines for axis in figure.axes)
    assert len(comparison_rows(artifacts)) == 24
    assert len(metrics_rows(artifacts)) == 6

    exports = export_comparison(artifacts, tmp_path / "exports")
    assert set(exports) == {
        "png",
        "svg",
        "force_by_object_png",
        "force_by_object_svg",
        "percentage_error_png",
        "percentage_error_svg",
        "data_csv",
        "metrics_csv",
        "statistics_csv",
    }
    assert all((tmp_path / "exports" / path.split("/")[-1]).is_file() for path in exports.values())


def test_single_gripper_calibration_has_six_panels():
    artifacts = {
        experiment: _single_artifact(index / 10)
        for index, experiment in enumerate(EXPERIMENT_IDS, start=1)
    }

    figure = calibration_figure(artifacts)

    assert len(figure.axes) == 6
    assert len(comparison_rows(artifacts)) == 12
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


def test_individual_benchmark_plot_orders_objects_and_adds_images(tmp_path):
    artifact = _single_artifact(0.25)
    artifact["metadata"]["experiment"] = "e3"
    artifact["rows"].reverse()
    for index, row in enumerate(artifact["rows"]):
        image_path = tmp_path / f"object_{index}.png"
        Image.new("RGB", (12, 8), color=(40 + index, 80, 120)).save(image_path)
        row["image_path"] = image_path.name

    figure = individual_calibration_figure(artifact, image_root=tmp_path)
    axis = figure.axes[0]

    assert list(axis.lines[0].get_ydata()) == [1.5, 2.5]
    assert list(axis.collections[0].get_offsets()[:, 1]) == [1.75, 2.75]
    assert axis.lines[0].get_label() == "Oracle minimum force"
    assert axis.collections[0].get_label() == "Predicted force"
    assert len(axis.artists) == 2
    assert axis.get_ylabel() == "Force (N)"


def test_suite_force_plot_overlays_all_experiments_with_stable_styles():
    artifacts = {
        experiment: _artifact(index / 10)
        for index, experiment in enumerate(EXPERIMENT_IDS, start=1)
    }

    figure = suite_force_by_object_figure(artifacts)

    assert len(figure.axes) == 2
    for axis in figure.axes:
        assert axis.lines[0].get_label() == "Oracle minimum force"
        assert [collection.get_label() for collection in axis.collections] == [
            experiment.upper() for experiment in EXPERIMENT_IDS
        ]
        for collection, experiment in zip(
            axis.collections,
            EXPERIMENT_IDS,
            strict=True,
        ):
            expected = EXPERIMENT_STYLES[experiment]["color"].lower()
            actual = collection.get_facecolor()[0]
            assert "#{:02x}{:02x}{:02x}".format(
                *(round(channel * 255) for channel in actual[:3])
            ) == expected


def test_suite_percentage_error_is_signed_and_omits_zero_truth():
    artifacts = {
        experiment: _artifact(index / 10)
        for index, experiment in enumerate(EXPERIMENT_IDS, start=1)
    }
    for artifact in artifacts.values():
        row = next(row for row in artifact["rows"] if row["object_id"] == "object_a")
        row["true_gecko_force_n"] = 0.0

    figure = suite_percentage_error_figure(artifacts)
    gecko_axis = figure.axes[0]

    assert gecko_axis.lines[0].get_label() == "Zero error"
    assert "1 zero-truth omitted" in gecko_axis.get_title()
    assert list(gecko_axis.collections[0].get_offsets()[:, 1]) == pytest.approx([5.0])
    assert list(gecko_axis.collections[1].get_offsets()[:, 1]) == pytest.approx([10.0])


def test_force_error_statistics_include_direction_and_active_gripper_scopes():
    artifacts = {
        experiment: _artifact(0.0) for experiment in EXPERIMENT_IDS
    }
    for artifact in artifacts.values():
        first, second = artifact["rows"]
        first["pred_gecko_force_n"] = 2.0
        second["pred_gecko_force_n"] = 1.5
        first["pred_silicone_force_n"] = 1.5
        second["pred_silicone_force_n"] = 4.5

    rows = force_error_statistics_rows(artifacts)
    e1_rows = {row["scope"]: row for row in rows if row["experiment"] == "E1"}

    assert set(e1_rows) == {"Overall", "Gecko", "Silicone"}
    gecko = e1_rows["Gecko"]
    assert gecko["n"] == 2
    assert gecko["mae_n"] == pytest.approx(0.75)
    assert gecko["rmse_n"] == pytest.approx((0.625) ** 0.5)
    assert gecko["residual_std_n"] == pytest.approx(1.5 / (2**0.5))
    assert gecko["overprediction_count"] == 1
    assert gecko["underprediction_count"] == 1
    assert gecko["exact_prediction_count"] == 0
    assert gecko["average_overprediction_n"] == pytest.approx(1.0)
    assert gecko["average_underprediction_n"] == pytest.approx(0.5)

    silicone = e1_rows["Silicone"]
    assert silicone["overprediction_count"] == 1
    assert silicone["underprediction_count"] == 0
    assert silicone["exact_prediction_count"] == 1
    assert silicone["average_overprediction_n"] == pytest.approx(2.0)
    assert silicone["average_underprediction_n"] is None

    single = {
        experiment: _single_artifact(index / 10)
        for index, experiment in enumerate(EXPERIMENT_IDS, start=1)
    }
    single_rows = force_error_statistics_rows(single)
    assert len(single_rows) == len(EXPERIMENT_IDS)
    assert {row["scope"] for row in single_rows} == {"Silicone"}
