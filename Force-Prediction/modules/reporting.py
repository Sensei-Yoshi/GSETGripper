"""Pure comparison-table, calibration-figure, and paper-export helpers."""

from __future__ import annotations

import copy
import csv
import math
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


def _benchmark_force_points(artifact: dict, gripper: str) -> list[dict]:
    """Return evaluable rows ordered by oracle minimum force, then object ID."""
    points = [
        row
        for row in artifact.get("rows", [])
        if row.get(f"true_{gripper}_force_n") is not None
        and row.get(f"pred_{gripper}_force_n") is not None
    ]
    return sorted(
        points,
        key=lambda row: (
            float(row[f"true_{gripper}_force_n"]),
            str(row.get("object_id", "")),
        ),
    )


def _add_object_thumbnail(axis, x: int, row: dict, image_root: Path | None) -> None:  # noqa: ANN001
    """Place a saved object image below its x position, with a text fallback."""
    image_path = row.get("image_path")
    if image_root is not None and image_path:
        path = Path(image_path)
        if not path.is_absolute():
            path = image_root / path
        try:
            from matplotlib.offsetbox import AnnotationBbox, OffsetImage
            from PIL import Image, ImageOps

            with Image.open(path) as source:
                source = ImageOps.exif_transpose(source).convert("RGB")
                source.thumbnail((72, 72))
                thumbnail = Image.new("RGB", (72, 72), "white")
                offset = (
                    (thumbnail.width - source.width) // 2,
                    (thumbnail.height - source.height) // 2,
                )
                thumbnail.paste(source, offset)
            artist = AnnotationBbox(
                OffsetImage(thumbnail, zoom=0.48),
                (x, -0.14),
                xycoords=("data", "axes fraction"),
                frameon=True,
                bboxprops={"edgecolor": "#d1d5db", "linewidth": 0.6},
                pad=0.05,
                box_alignment=(0.5, 1.0),
                annotation_clip=False,
            )
            axis.add_artist(artist)
            return
        except (OSError, ValueError):
            pass

    label = str(row.get("object_name") or row.get("object_id") or x)
    axis.text(
        x,
        -0.11,
        label.replace("_", " "),
        transform=axis.get_xaxis_transform(),
        ha="right",
        va="top",
        rotation=55,
        fontsize=7,
        color="#4b5563",
        clip_on=False,
    )


def individual_calibration_figure(
    artifact: dict,
    image_root: str | Path | None = None,
):  # noqa: ANN201
    """Plot predicted and oracle forces by force-ordered benchmark object."""
    plt = _pyplot()
    grippers = artifact_grippers(artifact)
    experiment = artifact.get("metadata", {}).get("experiment", "benchmark")
    resolved_image_root = Path(image_root) if image_root is not None else None
    point_sets = {
        gripper: _benchmark_force_points(artifact, gripper) for gripper in grippers
    }
    largest_count = max((len(points) for points in point_sets.values()), default=1)
    figure, axes = plt.subplots(
        1,
        len(grippers),
        figsize=(max(7.5, largest_count * 0.72) * len(grippers), 5.8),
        sharey=True,
        squeeze=False,
    )
    all_forces = [
        float(row[field])
        for gripper, points in point_sets.items()
        for row in points
        for field in (f"true_{gripper}_force_n", f"pred_{gripper}_force_n")
    ]
    y_max = max(1.0, math.ceil(max(all_forces, default=1.0) * 1.15 * 2) / 2)
    for index, gripper in enumerate(grippers):
        axis = axes[0][index]
        points = point_sets[gripper]
        positions = list(range(len(points)))
        truths = [float(row[f"true_{gripper}_force_n"]) for row in points]
        predictions = [float(row[f"pred_{gripper}_force_n"]) for row in points]
        axis.plot(
            positions,
            truths,
            color="#475569",
            linewidth=1.0,
            marker="o",
            markersize=5,
            markerfacecolor="#ffffff",
            markeredgewidth=1.4,
            label="Oracle minimum force",
            zorder=2,
        )
        axis.scatter(
            positions,
            predictions,
            s=38,
            color="#e76f51",
            marker="D",
            edgecolors="#ffffff",
            linewidths=0.7,
            label="Predicted force",
            zorder=3,
        )
        for x, row in zip(positions, points, strict=True):
            _add_object_thumbnail(axis, x, row, resolved_image_root)
        metrics = artifact.get("metrics", {}).get("force", {}).get(gripper, {})
        axis.set_title(
            f"{experiment.upper()} — {gripper.title()}\n"
            f"MAE {metrics.get('mae', float('nan')):.3f} N | "
            f"RMSE {metrics.get('rmse', float('nan')):.3f} N | "
            f"n={metrics.get('n', 0)}",
            fontsize=10,
        )
        axis.set_xlabel(
            "Objects ordered by ground-truth minimum force",
            labelpad=62,
        )
        if index == 0:
            axis.set_ylabel("Force (N)")
        axis.set_xlim(-0.55, max(len(points) - 0.45, 0.55))
        axis.set_ylim(0, y_max)
        axis.set_xticks(positions, labels=[])
        axis.tick_params(axis="x", length=0)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="upper left", frameon=False, fontsize=9)
    figure.suptitle("Benchmark force by object", fontsize=13)
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))
    return figure


def export_individual_evaluation(
    artifact: dict,
    destination: Path,
    stem: str,
    *,
    image_root: str | Path | None = None,
) -> dict[str, Path]:
    """Persist deterministic PNG/SVG plots for one benchmark evaluation."""
    destination.mkdir(parents=True, exist_ok=True)
    figure = individual_calibration_figure(artifact, image_root=image_root)
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
