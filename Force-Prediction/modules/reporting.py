"""Pure comparison-table, calibration-figure, and paper-export helpers."""

from __future__ import annotations

import copy
import csv
from pathlib import Path

from .config import Config
from .contracts import (
    ExperienceRecord,
    Gripper,
    PerGripperPrediction,
    SelectionResult,
    group_by_object,
)
from .evaluation import EvalRow, compute_metrics

PRIMARY_EXPERIMENTS = ("e1", "e2", "e3", "e4")
LEGACY_GRIPPERS = ("gecko", "silicone")


def artifact_grippers(artifact: dict) -> tuple[str, ...]:
    """Read target provenance, defaulting historical artifacts to paired."""
    metadata = artifact.get("metadata", artifact)
    names = metadata.get("active_grippers")
    if not names and artifact.get("rows"):
        names = artifact["rows"][0].get("active_grippers")
    return tuple(names or LEGACY_GRIPPERS)


def common_intersection_artifacts(
    artifacts: dict[str, dict],
    cfg: Config,
) -> tuple[dict[str, dict], tuple[str, ...]]:
    """Filter all four artifacts to the same objects and recompute comparable metrics."""
    if any(experiment not in artifacts for experiment in PRIMARY_EXPERIMENTS):
        return {}, ()
    target_sets = {
        artifact_grippers(artifacts[experiment]) for experiment in PRIMARY_EXPERIMENTS
    }
    if len(target_sets) != 1:
        raise ValueError("cross-experiment comparison requires identical active grippers")
    names = next(iter(target_sets))
    metrics_cfg = cfg.model_copy(deep=True)
    metrics_cfg.prediction.active_grippers = tuple(Gripper(name) for name in names)
    object_sets = [
        {row["object_id"] for row in artifacts[experiment]["rows"]}
        for experiment in PRIMARY_EXPERIMENTS
    ]
    common_ids = tuple(sorted(set.intersection(*object_sets))) if object_sets else ()
    filtered: dict[str, dict] = {}
    for experiment in PRIMARY_EXPERIMENTS:
        artifact = copy.deepcopy(artifacts[experiment])
        rows = [row for row in artifact["rows"] if row["object_id"] in common_ids]
        artifact["rows"] = rows
        artifact["metrics"] = _metrics_from_rows(rows, metrics_cfg, names)
        filtered[experiment] = artifact
    return filtered, common_ids


def _metrics_from_rows(
    rows: list[dict], cfg: Config, grippers: tuple[str, ...]
) -> dict:
    eval_rows: list[EvalRow] = []
    for row in rows:
        truth_records: list[ExperienceRecord] = []
        predictions: dict[str, PerGripperPrediction] = {}
        for name in grippers:
            feasible_value = row.get(f"true_{name}_feasible")
            feasible = (
                bool(feasible_value)
                if feasible_value is not None
                else row.get(f"true_{name}_force_n") is not None
            )
            truth_records.append(
                ExperienceRecord(
                    object_id=row["object_id"],
                    image_path="",
                    gripper=Gripper(name),
                    min_force_n=row.get(f"true_{name}_force_n") if feasible else None,
                    feasible=feasible,
                    failed_at_limit_n=None if feasible else cfg.force.limit_n,
                )
            )
            predictions[name] = PerGripperPrediction(
                candidate_gripper=Gripper(name),
                feasible=bool(row.get(f"pred_{name}_feasible", True)),
                predicted_normal_force_n=float(row[f"pred_{name}_force_n"]),
            )
        truth = group_by_object(truth_records)[row["object_id"]]
        selection = SelectionResult(
            desired_gripper=row["predicted_gripper"],
            predicted_normal_force_n=(
                predictions[row["predicted_gripper"]].predicted_normal_force_n
                if row["predicted_gripper"] in predictions
                else None
            ),
            candidate_predictions=predictions,
            model_recommended_gripper=row.get("model_recommended_gripper"),
        )
        eval_rows.append(EvalRow(object_id=row["object_id"], truth=truth, result=selection))
    return compute_metrics(eval_rows, cfg).to_dict()


def _pyplot():
    """Load matplotlib with a non-interactive backend for tests and exports."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def comparison_rows(artifacts: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for experiment in PRIMARY_EXPERIMENTS:
        artifact = artifacts.get(experiment)
        if artifact is None:
            continue
        for source in artifact["rows"]:
            for gripper in artifact_grippers(artifact):
                truth = source.get(f"true_{gripper}_force_n")
                predicted = source.get(f"pred_{gripper}_force_n")
                rows.append(
                    {
                        "experiment": experiment.upper(),
                        "experiment_id": experiment,
                        "object_id": source["object_id"],
                        "gripper": gripper,
                        "true_force_n": truth,
                        "predicted_force_n": predicted,
                        "signed_error_n": (
                            predicted - truth
                            if truth is not None and predicted is not None
                            else None
                        ),
                        "absolute_error_n": (
                            abs(predicted - truth)
                            if truth is not None and predicted is not None
                            else None
                        ),
                        "true_favored": source.get("true_favored"),
                        "predicted_gripper": source.get("predicted_gripper"),
                        "selection_correct": source.get("selection_correct"),
                        "regret_n": source.get("regret_n"),
                    }
                )
    return rows


def metrics_rows(artifacts: dict[str, dict]) -> list[dict]:
    rows = []
    for experiment in PRIMARY_EXPERIMENTS:
        artifact = artifacts.get(experiment)
        if artifact is None:
            continue
        metrics = artifact["metrics"]
        grippers = artifact_grippers(artifact)
        row = {
            "experiment": experiment.upper(),
            "active_grippers": ",".join(grippers),
        }
        for gripper in (*grippers, "overall"):
            force = metrics["force"].get(gripper, {"n": 0})
            for metric in ("n", "mae", "rmse", "medae", "within_0.25n", "within_0.5n"):
                row[f"{gripper}_{metric}"] = force.get(metric)
        selection = metrics["selection"]
        for metric in (
            "accuracy",
            "infeasible_pick_rate",
            "mean_regret_n",
            "median_regret_n",
            "worst_regret_n",
        ):
            row[f"selection_{metric}"] = selection.get(metric)
        recommendation = metrics["model_recommendation"]
        row["recommendation_accuracy"] = recommendation.get("accuracy")
        row["recommendation_selector_agreement"] = recommendation.get(
            "selector_agreement"
        )
        rows.append(row)
    return rows


def calibration_figure(artifacts: dict[str, dict]):  # noqa: ANN201
    """Return target-aware gripper/experiment calibration panels."""
    plt = _pyplot()

    available = [artifact for artifact in artifacts.values() if artifact is not None]
    target_sets = {artifact_grippers(artifact) for artifact in available}
    if len(target_sets) > 1:
        raise ValueError("calibration figure requires identical active grippers")
    grippers = next(iter(target_sets), LEGACY_GRIPPERS)
    long_rows = comparison_rows(artifacts)
    figure, axes = plt.subplots(
        len(grippers),
        4,
        figsize=(15, 3.75 * len(grippers)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    colors = {"gecko": "#147a4a", "silicone": "#b45f19"}
    for row_index, gripper in enumerate(grippers):
        for column_index, experiment in enumerate(PRIMARY_EXPERIMENTS):
            axis = axes[row_index][column_index]
            points = [
                row
                for row in long_rows
                if row["gripper"] == gripper
                and row["experiment_id"] == experiment
                and row["true_force_n"] is not None
                and row["predicted_force_n"] is not None
            ]
            axis.scatter(
                [row["true_force_n"] for row in points],
                [row["predicted_force_n"] for row in points],
                s=18,
                alpha=0.72,
                color=colors[gripper],
                edgecolors="none",
            )
            metrics = artifacts.get(experiment, {}).get("metrics", {}).get("force", {}).get(
                gripper, {}
            )
            axis.set_title(
                f"{experiment.upper()} — {gripper.title()}\n"
                f"MAE {metrics.get('mae', float('nan')):.3f} N | "
                f"RMSE {metrics.get('rmse', float('nan')):.3f} N | n={metrics.get('n', 0)}",
                fontsize=9,
            )
            axis.set_xlim(0, 8)
            axis.set_ylim(0, 8)
            axis.grid(alpha=0.18)
            if row_index == len(grippers) - 1:
                axis.set_xlabel("Ground-truth force (N)")
            if column_index == 0:
                axis.set_ylabel("Predicted force (N)")
    figure.suptitle("E1–E4 force calibration by gripper", fontsize=14)
    figure.tight_layout()
    return figure


def individual_calibration_figure(artifact: dict):  # noqa: ANN201
    """Return one calibration panel per active gripper for one evaluation."""
    plt = _pyplot()
    grippers = artifact_grippers(artifact)
    experiment = artifact.get("metadata", {}).get("experiment", "benchmark")
    figure, axes = plt.subplots(
        1,
        len(grippers),
        figsize=(5.25 * len(grippers), 4.5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    colors = {"gecko": "#147a4a", "silicone": "#b45f19"}
    for index, gripper in enumerate(grippers):
        axis = axes[0][index]
        points = [
            row
            for row in artifact.get("rows", [])
            if row.get(f"true_{gripper}_force_n") is not None
            and row.get(f"pred_{gripper}_force_n") is not None
        ]
        axis.scatter(
            [row[f"true_{gripper}_force_n"] for row in points],
            [row[f"pred_{gripper}_force_n"] for row in points],
            s=24,
            alpha=0.75,
            color=colors[gripper],
            edgecolors="none",
        )
        metrics = artifact.get("metrics", {}).get("force", {}).get(gripper, {})
        axis.set_title(
            f"{experiment.upper()} — {gripper.title()}\n"
            f"MAE {metrics.get('mae', float('nan')):.3f} N | "
            f"RMSE {metrics.get('rmse', float('nan')):.3f} N | "
            f"n={metrics.get('n', 0)}",
            fontsize=10,
        )
        axis.set_xlabel("Ground-truth force (N)")
        if index == 0:
            axis.set_ylabel("Predicted force (N)")
        axis.set_xlim(0, 8)
        axis.set_ylim(0, 8)
        axis.grid(alpha=0.18)
    figure.suptitle("Force calibration", fontsize=13)
    figure.tight_layout()
    return figure


def export_individual_evaluation(
    artifact: dict,
    destination: Path,
    stem: str,
) -> dict[str, Path]:
    """Persist deterministic PNG/SVG plots for one benchmark evaluation."""
    destination.mkdir(parents=True, exist_ok=True)
    figure = individual_calibration_figure(artifact)
    png = destination / f"{stem}.png"
    svg = destination / f"{stem}.svg"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(svg, bbox_inches="tight")
    plt = _pyplot()
    plt.close(figure)
    return {"png": png, "svg": svg}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_comparison(artifacts: dict[str, dict], destination: Path) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=True)
    figure = calibration_figure(artifacts)
    png = destination / "calibration_e1_e4.png"
    svg = destination / "calibration_e1_e4.svg"
    data_csv = destination / "calibration_e1_e4_data.csv"
    metrics_csv = destination / "metrics_e1_e4.csv"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(svg, bbox_inches="tight")
    plt = _pyplot()
    plt.close(figure)
    _write_csv(data_csv, comparison_rows(artifacts))
    _write_csv(metrics_csv, metrics_rows(artifacts))
    return {
        "png": str(png),
        "svg": str(svg),
        "data_csv": str(data_csv),
        "metrics_csv": str(metrics_csv),
    }
