"""Pure comparison-table, calibration-figure, and paper-export helpers."""

from __future__ import annotations

import copy
import csv
import math
from pathlib import Path

import numpy as np

from .config import Config
from .contracts import (
    ExperienceRecord,
    Gripper,
    PerGripperPrediction,
    SelectionResult,
    group_by_object,
)
from .evaluation import EvalRow, compute_metrics

PRIMARY_EXPERIMENTS = ("e1", "e2", "e3", "e4", "e5", "e6")
LEGACY_GRIPPERS = ("gecko", "silicone")
EXPERIMENT_STYLES = {
    "e1": {"color": "#0072B2", "marker": "o"},
    "e2": {"color": "#E69F00", "marker": "s"},
    "e3": {"color": "#009E73", "marker": "^"},
    "e4": {"color": "#CC79A7", "marker": "D"},
    "e5": {"color": "#D55E00", "marker": "v"},
    "e6": {"color": "#56B4E9", "marker": "P"},
}
ERROR_ZERO_TOLERANCE_N = 1e-9


def present_experiments(artifacts: dict[str, dict]) -> tuple[str, ...]:
    """Experiments actually present in a bundle, in canonical E1→E6 order.

    Suites may run any subset (e.g. E1–E5), so comparison and figure helpers
    iterate the experiments that are present rather than assuming all six.
    """
    return tuple(name for name in PRIMARY_EXPERIMENTS if artifacts.get(name) is not None)


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
    """Filter the present artifacts to shared objects and recompute comparable metrics."""
    present = present_experiments(artifacts)
    if not present:
        return {}, ()
    target_sets = {
        artifact_grippers(artifacts[experiment]) for experiment in present
    }
    if len(target_sets) != 1:
        raise ValueError("cross-experiment comparison requires identical active grippers")
    names = next(iter(target_sets))
    metrics_cfg = cfg.model_copy(deep=True)
    metrics_cfg.prediction.active_grippers = tuple(Gripper(name) for name in names)
    object_sets = [
        {row["object_id"] for row in artifacts[experiment]["rows"]}
        for experiment in present
    ]
    common_ids = tuple(sorted(set.intersection(*object_sets))) if object_sets else ()
    filtered: dict[str, dict] = {}
    for experiment in present:
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
                        "true_feasible": source.get(f"true_{gripper}_feasible"),
                        "predicted_feasible": source.get(
                            f"pred_{gripper}_feasible"
                        ),
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
                        "signed_percentage_error": (
                            100.0 * (predicted - truth) / truth
                            if truth is not None
                            and predicted is not None
                            and abs(truth) > ERROR_ZERO_TOLERANCE_N
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


def _target_grippers(artifacts: dict[str, dict]) -> tuple[str, ...]:
    """Return the shared active-gripper contract for a comparison."""
    available = [artifact for artifact in artifacts.values() if artifact is not None]
    target_sets = {artifact_grippers(artifact) for artifact in available}
    if len(target_sets) > 1:
        raise ValueError("suite comparison requires identical active grippers")
    return next(iter(target_sets), LEGACY_GRIPPERS)


def _suite_gripper_points(
    artifacts: dict[str, dict],
    gripper: str,
) -> list[dict]:
    """Align all experiment predictions to one truth-ordered object sequence."""
    present = present_experiments(artifacts)
    if not present:
        return []

    rows_by_experiment = {
        experiment: {
            str(row["object_id"]): row
            for row in artifacts[experiment].get("rows", [])
        }
        for experiment in present
    }
    common_ids = set.intersection(
        *(set(rows) for rows in rows_by_experiment.values())
    )
    points: list[dict] = []
    for object_id in common_ids:
        anchor = rows_by_experiment[present[0]][object_id]
        truth = anchor.get(f"true_{gripper}_force_n")
        if truth is None:
            continue
        predictions = {
            experiment: rows_by_experiment[experiment][object_id].get(
                f"pred_{gripper}_force_n"
            )
            for experiment in present
        }
        if any(value is None for value in predictions.values()):
            continue
        points.append(
            {
                "object_id": object_id,
                "object_name": anchor.get("object_name"),
                "image_path": anchor.get("image_path"),
                "true_force_n": float(truth),
                "predictions": {
                    experiment: float(value)
                    for experiment, value in predictions.items()
                    if value is not None
                },
            }
        )
    return sorted(
        points,
        key=lambda point: (point["true_force_n"], point["object_id"]),
    )


def _suite_point_sets(
    artifacts: dict[str, dict],
) -> tuple[tuple[str, ...], dict[str, list[dict]]]:
    grippers = _target_grippers(artifacts)
    return grippers, {
        gripper: _suite_gripper_points(artifacts, gripper) for gripper in grippers
    }


def _experiment_offsets(experiments: tuple[str, ...]) -> dict[str, float]:
    offsets = np.linspace(-0.24, 0.24, len(experiments))
    return dict(zip(experiments, offsets, strict=True))


def _suite_figure(
    point_sets: dict[str, list[dict]],
    grippers: tuple[str, ...],
    *,
    sharey: bool,
):  # noqa: ANN201
    plt = _pyplot()
    largest_count = max((len(points) for points in point_sets.values()), default=1)
    return plt.subplots(
        1,
        len(grippers),
        figsize=(max(7.5, largest_count * 0.72) * len(grippers), 6.2),
        sharey=sharey,
        squeeze=False,
    )


def _style_suite_axis(axis, positions: list[int], point_count: int) -> None:  # noqa: ANN001
    axis.set_xlabel("Objects ordered by ground-truth minimum force", labelpad=62)
    axis.set_xlim(-0.65, max(point_count - 0.35, 0.65))
    axis.set_xticks(positions, labels=[])
    axis.tick_params(axis="x", length=0)
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)


def suite_force_by_object_figure(
    artifacts: dict[str, dict],
    image_root: str | Path | None = None,
):  # noqa: ANN201
    """Overlay E1–E6 force predictions on one object plot per active gripper."""
    present = present_experiments(artifacts)
    grippers, point_sets = _suite_point_sets(artifacts)
    resolved_image_root = Path(image_root) if image_root is not None else None
    figure, axes = _suite_figure(point_sets, grippers, sharey=True)
    offsets = _experiment_offsets(present)
    all_forces = [
        force
        for points in point_sets.values()
        for point in points
        for force in (
            point["true_force_n"],
            *[point["predictions"][name] for name in present],
        )
    ]
    y_max = max(1.0, math.ceil(max(all_forces, default=1.0) * 1.15 * 2) / 2)

    for index, gripper in enumerate(grippers):
        axis = axes[0][index]
        points = point_sets[gripper]
        positions = list(range(len(points)))
        axis.plot(
            positions,
            [point["true_force_n"] for point in points],
            color="#475569",
            linewidth=1.1,
            marker="o",
            markersize=5,
            markerfacecolor="#ffffff",
            markeredgewidth=1.4,
            label="Oracle minimum force",
            zorder=2,
        )
        for experiment in present:
            style = EXPERIMENT_STYLES[experiment]
            axis.scatter(
                [position + offsets[experiment] for position in positions],
                [point["predictions"][experiment] for point in points],
                s=38,
                color=style["color"],
                marker=style["marker"],
                edgecolors="#ffffff",
                linewidths=0.65,
                label=experiment.upper(),
                zorder=3,
            )
        for x, point in zip(positions, points, strict=True):
            _add_object_thumbnail(axis, x, point, resolved_image_root)
        axis.set_title(f"{gripper.title()} — {len(points)} objects", fontsize=10)
        if index == 0:
            axis.set_ylabel("Force (N)")
        axis.set_ylim(0, y_max)
        _style_suite_axis(axis, positions, len(points))

    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(present) + 1,
        frameon=False,
        fontsize=9,
    )
    figure.suptitle("Suite force predictions by object", fontsize=13)
    figure.tight_layout(rect=(0, 0.08, 1, 0.88))
    return figure


def suite_percentage_error_figure(
    artifacts: dict[str, dict],
    image_root: str | Path | None = None,
):  # noqa: ANN201
    """Overlay signed E1–E6 percentage errors per active gripper and object."""
    present = present_experiments(artifacts)
    grippers, point_sets = _suite_point_sets(artifacts)
    resolved_image_root = Path(image_root) if image_root is not None else None
    figure, axes = _suite_figure(point_sets, grippers, sharey=True)
    offsets = _experiment_offsets(present)
    all_percentages: list[float] = []
    total_omitted = 0

    for index, gripper in enumerate(grippers):
        axis = axes[0][index]
        points = point_sets[gripper]
        positions = list(range(len(points)))
        omitted = sum(
            abs(point["true_force_n"]) <= ERROR_ZERO_TOLERANCE_N
            for point in points
        )
        total_omitted += omitted
        axis.axhline(
            0.0,
            color="#475569",
            linewidth=1.1,
            linestyle="--",
            label="Zero error",
            zorder=1,
        )
        for experiment in present:
            plotted = [
                (position, point)
                for position, point in zip(positions, points, strict=True)
                if abs(point["true_force_n"]) > ERROR_ZERO_TOLERANCE_N
            ]
            percentages = [
                100.0
                * (point["predictions"][experiment] - point["true_force_n"])
                / point["true_force_n"]
                for _, point in plotted
            ]
            all_percentages.extend(percentages)
            style = EXPERIMENT_STYLES[experiment]
            axis.scatter(
                [position + offsets[experiment] for position, _ in plotted],
                percentages,
                s=38,
                color=style["color"],
                marker=style["marker"],
                edgecolors="#ffffff",
                linewidths=0.65,
                label=experiment.upper(),
                zorder=3,
            )
        for x, point in zip(positions, points, strict=True):
            _add_object_thumbnail(axis, x, point, resolved_image_root)
        omission = f" | {omitted} zero-truth omitted" if omitted else ""
        axis.set_title(
            f"{gripper.title()} — {len(points) - omitted} objects{omission}",
            fontsize=10,
        )
        if index == 0:
            axis.set_ylabel("Signed percentage error (%)")
        _style_suite_axis(axis, positions, len(points))

    percentage_limit = max(
        10.0,
        math.ceil(max((abs(value) for value in all_percentages), default=0.0) * 1.15 / 10)
        * 10,
    )
    for axis in axes[0]:
        axis.set_ylim(-percentage_limit, percentage_limit)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(present) + 1,
        frameon=False,
        fontsize=9,
    )
    subtitle = "Suite signed percentage error by object"
    if total_omitted:
        subtitle += f" — {total_omitted} zero-truth object/gripper cases omitted"
    figure.suptitle(subtitle, fontsize=13)
    figure.tight_layout(rect=(0, 0.08, 1, 0.88))
    return figure


def _scoreable_residuals(artifact: dict, gripper: str) -> list[float]:
    residuals: list[float] = []
    for row in artifact.get("rows", []):
        truth = row.get(f"true_{gripper}_force_n")
        predicted = row.get(f"pred_{gripper}_force_n")
        true_feasible = row.get(f"true_{gripper}_feasible")
        predicted_feasible = row.get(f"pred_{gripper}_feasible", True)
        if true_feasible is None:
            true_feasible = truth is not None
        if (
            truth is None
            or predicted is None
            or not true_feasible
            or not predicted_feasible
        ):
            continue
        residuals.append(float(predicted) - float(truth))
    return residuals


def _force_error_statistics(
    experiment: str,
    scope: str,
    residuals: list[float],
) -> dict:
    errors = np.asarray(residuals, dtype=float)
    over = errors[errors > ERROR_ZERO_TOLERANCE_N]
    under = errors[errors < -ERROR_ZERO_TOLERANCE_N]
    exact_count = int(len(errors) - len(over) - len(under))
    return {
        "experiment": experiment.upper(),
        "scope": scope,
        "n": int(len(errors)),
        "mae_n": float(np.mean(np.abs(errors))) if len(errors) else None,
        "rmse_n": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
        "residual_std_n": (
            float(np.std(errors, ddof=1)) if len(errors) >= 2 else None
        ),
        "overprediction_count": int(len(over)),
        "underprediction_count": int(len(under)),
        "exact_prediction_count": exact_count,
        "average_overprediction_n": float(np.mean(over)) if len(over) else None,
        "average_underprediction_n": (
            float(np.mean(np.abs(under))) if len(under) else None
        ),
    }


def force_error_statistics_rows(artifacts: dict[str, dict]) -> list[dict]:
    """Return experiment-level force-error statistics for active target scopes."""
    grippers = _target_grippers(artifacts)
    rows: list[dict] = []
    for experiment in PRIMARY_EXPERIMENTS:
        artifact = artifacts.get(experiment)
        if artifact is None:
            continue
        residuals = {
            gripper: _scoreable_residuals(artifact, gripper)
            for gripper in grippers
        }
        if len(grippers) > 1:
            rows.append(
                _force_error_statistics(
                    experiment,
                    "Overall",
                    [value for gripper in grippers for value in residuals[gripper]],
                )
            )
        rows.extend(
            _force_error_statistics(experiment, gripper.title(), residuals[gripper])
            for gripper in grippers
        )
    return rows


def calibration_figure(artifacts: dict[str, dict]):  # noqa: ANN201
    """Return target-aware gripper/experiment calibration panels."""
    plt = _pyplot()

    grippers = _target_grippers(artifacts)
    present = present_experiments(artifacts) or PRIMARY_EXPERIMENTS
    long_rows = comparison_rows(artifacts)
    figure, axes = plt.subplots(
        len(grippers),
        len(present),
        figsize=(3.75 * len(present), 3.75 * len(grippers)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    colors = {"gecko": "#147a4a", "silicone": "#b45f19"}
    for row_index, gripper in enumerate(grippers):
        for column_index, experiment in enumerate(present):
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
    figure.suptitle("E1–E6 force calibration by gripper", fontsize=14)
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

            with Image.open(path) as opened:
                source = ImageOps.exif_transpose(opened).convert("RGB")
                source.thumbnail((72, 72))
                thumbnail = Image.new("RGB", (72, 72), "white")
                offset = (
                    (thumbnail.width - source.width) // 2,
                    (thumbnail.height - source.height) // 2,
                )
                thumbnail.paste(source, offset)
            artist = AnnotationBbox(
                OffsetImage(np.asarray(thumbnail), zoom=0.48),
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


def export_comparison(
    artifacts: dict[str, dict],
    destination: Path,
    *,
    image_root: str | Path | None = None,
) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=True)
    calibration = calibration_figure(artifacts)
    force_by_object = suite_force_by_object_figure(
        artifacts,
        image_root=image_root,
    )
    percentage_error = suite_percentage_error_figure(
        artifacts,
        image_root=image_root,
    )
    png = destination / "calibration_e1_e6.png"
    svg = destination / "calibration_e1_e6.svg"
    force_png = destination / "force_by_object_e1_e6.png"
    force_svg = destination / "force_by_object_e1_e6.svg"
    percentage_png = destination / "percentage_error_e1_e6.png"
    percentage_svg = destination / "percentage_error_e1_e6.svg"
    data_csv = destination / "calibration_e1_e6_data.csv"
    metrics_csv = destination / "metrics_e1_e6.csv"
    statistics_csv = destination / "force_error_statistics_e1_e6.csv"
    calibration.savefig(png, dpi=300, bbox_inches="tight")
    calibration.savefig(svg, bbox_inches="tight")
    force_by_object.savefig(force_png, dpi=300, bbox_inches="tight")
    force_by_object.savefig(force_svg, bbox_inches="tight")
    percentage_error.savefig(percentage_png, dpi=300, bbox_inches="tight")
    percentage_error.savefig(percentage_svg, bbox_inches="tight")
    plt = _pyplot()
    plt.close(calibration)
    plt.close(force_by_object)
    plt.close(percentage_error)
    _write_csv(data_csv, comparison_rows(artifacts))
    _write_csv(metrics_csv, metrics_rows(artifacts))
    _write_csv(statistics_csv, force_error_statistics_rows(artifacts))
    return {
        "png": str(png),
        "svg": str(svg),
        "force_by_object_png": str(force_png),
        "force_by_object_svg": str(force_svg),
        "percentage_error_png": str(percentage_png),
        "percentage_error_svg": str(percentage_svg),
        "data_csv": str(data_csv),
        "metrics_csv": str(metrics_csv),
        "statistics_csv": str(statistics_csv),
    }
