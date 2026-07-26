"""Pure comparison-table, calibration-figure, and paper-export helpers."""

from __future__ import annotations

import csv
from pathlib import Path

PRIMARY_EXPERIMENTS = ("e1", "e2", "e3", "e4")
GRIPPERS = ("gecko", "silicone")


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
            for gripper in GRIPPERS:
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
        row = {"experiment": experiment.upper()}
        for gripper in (*GRIPPERS, "overall"):
            force = metrics["force"][gripper]
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
    """Return 2x4 gripper/experiment calibration panels without an identity line."""
    plt = _pyplot()

    long_rows = comparison_rows(artifacts)
    figure, axes = plt.subplots(2, 4, figsize=(15, 7.5), sharex=True, sharey=True)
    colors = {"gecko": "#147a4a", "silicone": "#b45f19"}
    for row_index, gripper in enumerate(GRIPPERS):
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
            if row_index == 1:
                axis.set_xlabel("Ground-truth force (N)")
            if column_index == 0:
                axis.set_ylabel("Predicted force (N)")
    figure.suptitle("E1–E4 force calibration by gripper", fontsize=14)
    figure.tight_layout()
    return figure


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
