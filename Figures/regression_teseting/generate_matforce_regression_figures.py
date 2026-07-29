"""Compare condition-level regressions for MatForceFinal and plot diagnostics.

Primary accuracy uses grouped outer and inner validation by physical surface.
Condition interpolation is reported separately by holding out only one condition.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    LeaveOneOut,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

FIGURES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FIGURES_DIR.parents[1] / "Force-Prediction"
DATASET_PATH = PROJECT_ROOT / "data" / "MatForceFinal" / "dataset.csv"

# Keep Matplotlib's cache out of the source tree on sandboxed systems.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matforce-matplotlib-cache")
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

BLUE = "#1769aa"
ORANGE = "#d97706"
DARK = "#202124"
MUTED = "#5f6368"
GRID = "#d7dce2"
LIGHT_BLUE = "#9ec5e5"

SVR_PARAMETERS = {
    "model__C": [1.0, 10.0, 30.0],
    "model__gamma": [0.03, 0.1, 0.3],
    "model__epsilon": [0.1, 0.4],
}


@dataclass(frozen=True)
class Observation:
    number: int
    name: str
    surface_id: str
    condition_id: str
    mass_g: float
    roughness_index: float
    projected_contact_fraction: float | None
    force_n: float


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    inputs: str
    columns: tuple[int, ...]
    estimator: Any


@dataclass
class ModelResult:
    spec: ModelSpec
    train_predictions: np.ndarray
    loo_predictions: np.ndarray
    fitted_estimator: Any
    train_r2: float
    loo_r2: float
    loo_rmse_n: float
    loo_mae_n: float
    interpolation_predictions: np.ndarray
    interpolation_r2: float
    interpolation_rmse_n: float
    interpolation_mae_n: float
    sample_count: int


def _optional_float(value: str | None) -> float | None:
    return float(value) if value and value.strip() else None


def _load_data() -> tuple[list[Observation], list[str]]:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Missing MatForceFinal dataset: {DATASET_PATH}")

    with DATASET_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    required = {"Object", "Mass_g", "roughness_index", "gecko_force_n"}
    missing = required.difference(rows[0] if rows else ())
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")

    complete: list[tuple[str, str, str, float, float, float | None, float]] = []
    excluded: list[str] = []
    for row in rows:
        name = row["Object"].strip()
        surface_id = "_".join(
            part for part in "".join(
                character.lower() if character.isalnum() else " " for character in name
            ).split()
        )
        condition_id = (row.get("condition_id") or "baseline").strip() or "baseline"
        mass_g = _optional_float(row.get("Mass_g"))
        roughness = _optional_float(row.get("roughness_index"))
        force_n = _optional_float(row.get("gecko_force_n"))
        contact = _optional_float(row.get("projected_contact_fraction"))
        if mass_g is None or roughness is None or force_n is None:
            excluded.append(name)
            continue
        if mass_g <= 0:
            raise ValueError(f"{name}: mass must be positive, got {mass_g}")
        if roughness < 0:
            raise ValueError(f"{name}: roughness must be nonnegative, got {roughness}")
        if not 0 <= force_n <= 8:
            raise ValueError(f"{name}: Gecko force must be within 0–8 N, got {force_n}")
        if contact is not None and not 0 <= contact <= 1:
            raise ValueError(f"{name}: projected contact fraction must be within 0–1")
        display_name = name if condition_id == "baseline" else f"{name} ({condition_id})"
        complete.append(
            (display_name, surface_id, condition_id, mass_g, roughness, contact, force_n)
        )

    complete.sort(key=lambda item: (item[1], item[2] != "baseline", item[2]))
    observations = [
        Observation(index, name, surface_id, condition_id, mass_g, roughness, contact, force_n)
        for index, (
            name, surface_id, condition_id, mass_g, roughness, contact, force_n
        ) in enumerate(complete, start=1)
    ]
    if len(observations) < 10:
        raise ValueError("At least 10 complete observations are required for comparison")
    return observations, excluded


def _set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "savefig.facecolor": "white",
        }
    )


def _save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _make_grid_search(estimator: Any, parameters: dict[str, list[Any]]) -> Any:
    return GridSearchCV(
        estimator,
        parameters,
        cv=LeaveOneOut(),
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )


def _model_specs(*, include_contact: bool) -> list[ModelSpec]:
    linear = Pipeline(
        [("scale", StandardScaler()), ("model", LinearRegression())]
    )
    quadratic_ridge = _make_grid_search(
        Pipeline(
            [
                ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
                ("scale", StandardScaler()),
                ("model", Ridge()),
            ]
        ),
        {"model__alpha": list(np.logspace(-3, 3, 9))},
    )
    rbf_svr = _make_grid_search(
        Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="rbf"))]),
        SVR_PARAMETERS,
    )
    specs = [
        ModelSpec(
            "mean_baseline",
            "Mean baseline",
            "none",
            (0, 1),
            DummyRegressor(strategy="mean"),
        ),
        ModelSpec(
            "roughness_linear",
            "Roughness-only linear",
            "roughness",
            (1,),
            clone(linear),
        ),
        ModelSpec(
            "mass_linear",
            "Log-mass-only linear",
            "log10(mass)",
            (0,),
            clone(linear),
        ),
        ModelSpec(
            "two_feature_linear",
            "Two-feature linear",
            "log10(mass) + roughness",
            (0, 1),
            clone(linear),
        ),
        ModelSpec(
            "two_feature_quadratic",
            "Two-feature quadratic ridge",
            "log10(mass) + roughness",
            (0, 1),
            quadratic_ridge,
        ),
        ModelSpec(
            "mass_rbf",
            "Log-mass-only RBF SVR",
            "log10(mass)",
            (0,),
            clone(rbf_svr),
        ),
        ModelSpec(
            "two_feature_rbf",
            "Two-feature RBF SVR",
            "log10(mass) + roughness",
            (0, 1),
            clone(rbf_svr),
        ),
    ]
    if include_contact:
        specs.extend(
            [
                ModelSpec(
                    "three_feature_linear",
                    "Mass + roughness + contact linear",
                    "log10(mass) + roughness + contact",
                    (0, 1, 2),
                    clone(linear),
                ),
                ModelSpec(
                    "three_feature_quadratic",
                    "Mass + roughness + contact quadratic ridge",
                    "log10(mass) + roughness + contact",
                    (0, 1, 2),
                    clone(quadratic_ridge),
                ),
                ModelSpec(
                    "three_feature_rbf",
                    "Mass + roughness + contact RBF SVR",
                    "log10(mass) + roughness + contact",
                    (0, 1, 2),
                    clone(rbf_svr),
                ),
            ]
        )
    return specs


def _evaluate_models(
    features: np.ndarray,
    targets: np.ndarray,
    surface_groups: np.ndarray,
    *,
    include_contact: bool,
) -> list[ModelResult]:
    results: list[ModelResult] = []
    for spec in _model_specs(include_contact=include_contact):
        selected = features[:, spec.columns]
        mask = np.all(np.isfinite(selected), axis=1)
        model_features = selected[mask]
        model_targets = targets[mask]
        model_groups = surface_groups[mask]
        if len(model_targets) < 3 or len(set(model_groups)) < 2:
            empty = np.full(len(targets), np.nan)
            results.append(
                ModelResult(
                    spec=spec,
                    train_predictions=empty.copy(),
                    loo_predictions=empty.copy(),
                    fitted_estimator=None,
                    train_r2=float("nan"),
                    loo_r2=float("nan"),
                    loo_rmse_n=float("nan"),
                    loo_mae_n=float("nan"),
                    interpolation_predictions=empty.copy(),
                    interpolation_r2=float("nan"),
                    interpolation_rmse_n=float("nan"),
                    interpolation_mae_n=float("nan"),
                    sample_count=int(mask.sum()),
                )
            )
            continue
        grouped_predictions = _nested_predictions(
            spec.estimator,
            model_features,
            model_targets,
            model_groups,
            grouped_outer=True,
        )
        interpolation_predictions = _nested_predictions(
            spec.estimator,
            model_features,
            model_targets,
            model_groups,
            grouped_outer=False,
        )
        fitted = _fit_model(
            clone(spec.estimator), model_features, model_targets, model_groups, grouped=True
        )
        fitted_values = np.clip(fitted.predict(model_features), 0, 8)
        train_predictions = np.full(len(targets), np.nan)
        loo_predictions = np.full(len(targets), np.nan)
        interpolation_full = np.full(len(targets), np.nan)
        train_predictions[mask] = fitted_values
        loo_predictions[mask] = grouped_predictions
        interpolation_full[mask] = interpolation_predictions
        results.append(
            ModelResult(
                spec=spec,
                train_predictions=train_predictions,
                loo_predictions=loo_predictions,
                fitted_estimator=fitted,
                train_r2=float(r2_score(model_targets, fitted_values)),
                loo_r2=float(r2_score(model_targets, grouped_predictions)),
                loo_rmse_n=float(
                    mean_squared_error(model_targets, grouped_predictions) ** 0.5
                ),
                loo_mae_n=float(mean_absolute_error(model_targets, grouped_predictions)),
                interpolation_predictions=interpolation_full,
                interpolation_r2=float(
                    r2_score(model_targets, interpolation_predictions)
                ),
                interpolation_rmse_n=float(
                    mean_squared_error(model_targets, interpolation_predictions) ** 0.5
                ),
                interpolation_mae_n=float(
                    mean_absolute_error(model_targets, interpolation_predictions)
                ),
                sample_count=int(mask.sum()),
            )
        )
    return results


def _fit_model(
    estimator: Any,
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    *,
    grouped: bool,
) -> Any:
    if isinstance(estimator, GridSearchCV):
        if grouped:
            unique_groups = len(set(groups))
            if unique_groups >= 2:
                estimator.cv = GroupKFold(n_splits=min(5, unique_groups))
                return estimator.fit(features, targets, groups=groups)
        estimator.cv = KFold(n_splits=min(5, len(targets)), shuffle=False)
    return estimator.fit(features, targets)


def _nested_predictions(
    estimator: Any,
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    *,
    grouped_outer: bool,
) -> np.ndarray:
    splitter = LeaveOneGroupOut() if grouped_outer else LeaveOneOut()
    split_groups = groups if grouped_outer else None
    predictions = np.empty(len(targets), dtype=float)
    for train_indices, test_indices in splitter.split(
        features, targets, groups=split_groups
    ):
        fitted = _fit_model(
            clone(estimator),
            features[train_indices],
            targets[train_indices],
            groups[train_indices],
            grouped=grouped_outer,
        )
        predictions[test_indices] = np.clip(
            fitted.predict(features[test_indices]), 0, 8
        )
    return predictions


def _selected_parameters(estimator: Any) -> dict[str, Any]:
    if not hasattr(estimator, "best_params_"):
        return {}
    return {
        name: float(value) if isinstance(value, np.floating) else value
        for name, value in estimator.best_params_.items()
    }


def _parameter_summary(estimator: Any) -> str:
    parameters = _selected_parameters(estimator)
    if not parameters:
        return "no tuned hyperparameters"
    formatted = []
    for name, value in parameters.items():
        short_name = name.removeprefix("model__")
        shown_value = f"{value:.3g}" if isinstance(value, float) else str(value)
        formatted.append(f"{short_name}={shown_value}")
    return ", ".join(formatted)


def _best_two_feature_result(results: list[ModelResult]) -> ModelResult:
    candidates = [
        result
        for result in results
        if result.spec.inputs == "log10(mass) + roughness"
    ]
    if not candidates:
        raise RuntimeError("No two-feature model results are available")
    return max(candidates, key=lambda result: result.loo_r2)


def _write_results(
    observations: list[Observation], results: list[ModelResult]
) -> None:
    metrics_path = FIGURES_DIR / "matforce_regression_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "inputs",
                "train_r2",
                "new_surface_grouped_r2",
                "new_surface_grouped_rmse_n",
                "new_surface_grouped_mae_n",
                "condition_interpolation_r2",
                "condition_interpolation_rmse_n",
                "condition_interpolation_mae_n",
                "sample_count",
                "full_data_selected_parameters",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "model": result.spec.label,
                    "inputs": result.spec.inputs,
                    "train_r2": f"{result.train_r2:.6f}",
                    "new_surface_grouped_r2": f"{result.loo_r2:.6f}",
                    "new_surface_grouped_rmse_n": f"{result.loo_rmse_n:.6f}",
                    "new_surface_grouped_mae_n": f"{result.loo_mae_n:.6f}",
                    "condition_interpolation_r2": f"{result.interpolation_r2:.6f}",
                    "condition_interpolation_rmse_n": (
                        f"{result.interpolation_rmse_n:.6f}"
                    ),
                    "condition_interpolation_mae_n": (
                        f"{result.interpolation_mae_n:.6f}"
                    ),
                    "sample_count": result.sample_count,
                    "full_data_selected_parameters": json.dumps(
                        _selected_parameters(result.fitted_estimator), sort_keys=True
                    ),
                }
            )

    predictions_path = FIGURES_DIR / "matforce_regression_loo_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Object",
                "surface_id",
                "condition_id",
                "Mass_g",
                "roughness_index",
                "ground_truth_force_n",
                "mass_only_rbf_loo_prediction_n",
                "best_two_feature_model",
                "best_two_feature_loo_prediction_n",
                "best_two_feature_residual_n",
            ],
        )
        writer.writeheader()
        by_key = {result.spec.key: result for result in results}
        if "mass_rbf" not in by_key:
            raise RuntimeError("Required mass-only RBF result is missing")
        best_two_feature = _best_two_feature_result(results)
        mass_predictions = by_key["mass_rbf"].loo_predictions
        combined_predictions = best_two_feature.loo_predictions
        for item, mass_prediction, combined_prediction in zip(
            observations, mass_predictions, combined_predictions, strict=True
        ):
            writer.writerow(
                {
                    "Object": item.name,
                    "surface_id": item.surface_id,
                    "condition_id": item.condition_id,
                    "Mass_g": f"{item.mass_g:.6g}",
                    "roughness_index": f"{item.roughness_index:.6g}",
                    "ground_truth_force_n": f"{item.force_n:.6g}",
                    "mass_only_rbf_loo_prediction_n": f"{mass_prediction:.6f}",
                    "best_two_feature_model": best_two_feature.spec.label,
                    "best_two_feature_loo_prediction_n": f"{combined_prediction:.6f}",
                    "best_two_feature_residual_n": (
                        f"{item.force_n - combined_prediction:.6f}"
                    ),
                }
            )


def _finish_axes(ax: plt.Axes) -> None:
    ax.grid(color=GRID, linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def _add_object_key(ax: plt.Axes, observations: list[Observation]) -> None:
    ax.set_axis_off()
    ax.set_title("Object key", loc="left", pad=10, color=DARK)
    rows_per_column = (len(observations) + 1) // 2
    x_positions = (0.0, 0.51)
    row_step = 0.92 / max(rows_per_column - 1, 1)
    for index, item in enumerate(observations):
        column = index // rows_per_column
        row = index % rows_per_column
        x = x_positions[column]
        y = 0.965 - row * row_step
        ax.text(
            x,
            y,
            f"{item.number:>2}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            color=BLUE,
            fontsize=8.2,
            fontweight="bold",
        )
        ax.text(
            x + 0.07,
            y,
            item.name,
            transform=ax.transAxes,
            ha="left",
            va="center",
            color=DARK,
            fontsize=7.8,
        )


def _plot_numbered_points(
    ax: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    observations: list[Observation],
    *,
    color: str,
) -> None:
    ax.scatter(
        x_values,
        y_values,
        s=42,
        color=color,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    fig = ax.figure
    fig.canvas.draw()
    display_points = ax.transData.transform(np.column_stack((x_values, y_values)))
    nearest = []
    for index, point in enumerate(display_points):
        distances = np.linalg.norm(display_points - point, axis=1)
        distances[index] = np.inf
        nearest.append(float(distances.min()))
    order = np.argsort(nearest)
    point_boxes = [
        Bbox.from_bounds(point[0] - 4, point[1] - 4, 8, 8)
        for point in display_points
    ]
    axes_box = ax.get_window_extent()
    label_boxes: list[Bbox] = []
    candidate_offsets = (
        (0, 9),
        (9, 7),
        (-9, 7),
        (11, 0),
        (-11, 0),
        (9, -8),
        (-9, -8),
        (0, -10),
        (16, 9),
        (-16, 9),
        (17, -8),
        (-17, -8),
        (0, 17),
        (0, -18),
    )

    for index in order:
        item = observations[index]
        annotation = ax.annotate(
            str(item.number),
            xy=(x_values[index], y_values[index]),
            xytext=candidate_offsets[0],
            textcoords="offset points",
            ha="center",
            va="center",
            color=color,
            fontsize=6.0,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.5,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "linewidth": 0.4,
                "shrinkA": 2,
                "shrinkB": 3,
            },
            zorder=4,
        )
        best_offset = candidate_offsets[0]
        best_box: Bbox | None = None
        best_score = np.inf
        for offset in candidate_offsets:
            annotation.set_position(offset)
            fig.canvas.draw()
            box = annotation.get_window_extent()
            overlap = sum(_overlap_area(box, other) for other in label_boxes)
            overlap += 0.35 * sum(
                _overlap_area(box, point_box)
                for point_index, point_box in enumerate(point_boxes)
                if point_index != index
            )
            outside = (
                max(0.0, axes_box.x0 - box.x0)
                + max(0.0, box.x1 - axes_box.x1)
                + max(0.0, axes_box.y0 - box.y0)
                + max(0.0, box.y1 - axes_box.y1)
            )
            distance_penalty = 0.01 * (offset[0] ** 2 + offset[1] ** 2)
            score = overlap + 250.0 * outside + distance_penalty
            if score < best_score:
                best_offset = offset
                best_box = box
                best_score = score
        annotation.set_position(best_offset)
        fig.canvas.draw()
        label_boxes.append(best_box or annotation.get_window_extent())


def _overlap_area(left: Bbox, right: Bbox) -> float:
    width = max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0))
    height = max(0.0, min(left.y1, right.y1) - max(left.y0, right.y0))
    return width * height


def _make_model_comparison_figure(
    results: list[ModelResult], excluded: list[str]
) -> None:
    best_two_feature = _best_two_feature_result(results)
    available = [result for result in results if np.isfinite(result.loo_r2)]
    ordered = sorted(available, key=lambda result: result.loo_r2)
    labels = [result.spec.label for result in ordered]
    y_positions = np.arange(len(ordered))
    train_scores = np.asarray([result.train_r2 for result in ordered])
    loo_scores = np.asarray([result.loo_r2 for result in ordered])

    fig, ax = plt.subplots(figsize=(10.8, 5.8), facecolor="white")
    fig.subplots_adjust(left=0.29, right=0.96, top=0.86, bottom=0.18)
    ax.hlines(y_positions, loo_scores, train_scores, color=GRID, linewidth=2, zorder=1)
    ax.scatter(
        train_scores,
        y_positions,
        s=58,
        facecolor="white",
        edgecolor=MUTED,
        linewidth=1.2,
        label="Training R² (optimistic)",
        zorder=2,
    )
    colors = [
        ORANGE if result.spec.key == "mass_rbf" else BLUE
        if result.spec.key == best_two_feature.spec.key
        else MUTED
        for result in ordered
    ]
    ax.scatter(
        loo_scores,
        y_positions,
        s=65,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        label="Grouped new-surface R²",
        zorder=3,
    )
    for y_position, train_score, loo_score in zip(
        y_positions, train_scores, loo_scores, strict=True
    ):
        ax.annotate(
            f"{train_score:.2f}",
            (train_score, y_position),
            xytext=(6, 7),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=7.5,
            color=MUTED,
        )
        ax.annotate(
            f"{loo_score:.2f}",
            (loo_score, y_position),
            xytext=(6, -8),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=7.5,
            color=DARK,
        )

    ax.axvline(0, color=DARK, linewidth=0.9, zorder=0)
    ax.set_xlim(-0.25, 0.82)
    ax.set_yticks(y_positions, labels)
    ax.set_xlabel("R² (higher is better; 1.0 is perfect)")
    ax.set_title("Regression comparison: performance on unseen physical surfaces")
    _finish_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.text(
        0.29,
        0.055,
        "Outer and inner validation are grouped by physical surface. "
        "Predictions are constrained to 0–8 N. "
        f"Excluded incomplete labels: {', '.join(excluded) if excluded else 'none'}.",
        ha="left",
        va="bottom",
        fontsize=7.8,
        color=MUTED,
    )
    _save_figure(fig, "matforce_regression_model_comparison")


def _make_prediction_figure(
    observations: list[Observation], results: list[ModelResult], excluded: list[str]
) -> None:
    by_key = {result.spec.key: result for result in results}
    selected = [by_key["mass_rbf"], _best_two_feature_result(results)]
    actual = np.asarray([item.force_n for item in observations])

    fig = plt.figure(figsize=(13.8, 6.3), facecolor="white")
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(1, 1, 1.15),
        left=0.06,
        right=0.985,
        top=0.86,
        bottom=0.15,
        wspace=0.22,
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    key_ax = fig.add_subplot(grid[0, 2])

    for ax, result, color in zip(axes, selected, (ORANGE, BLUE), strict=True):
        ax.plot([-0.2, 8.2], [-0.2, 8.2], color=MUTED, linestyle="--", linewidth=1.1)
        ax.set_xlim(-0.2, 8.0)
        ax.set_ylim(-0.2, 8.0)
        ax.set_aspect("equal", adjustable="box")
        _plot_numbered_points(
            ax,
            actual,
            result.loo_predictions,
            observations,
            color=color,
        )
        ax.set_xlabel("Ground-truth minimum Gecko force (N)")
        ax.set_title(result.spec.label)
        ax.text(
            0.04,
            0.96,
            f"Grouped R² = {result.loo_r2:.2f}\n"
            f"RMSE = {result.loo_rmse_n:.2f} N\n"
            f"MAE = {result.loo_mae_n:.2f} N",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            color=DARK,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": GRID,
                "alpha": 0.92,
            },
            zorder=5,
        )
        _finish_axes(ax)
    axes[0].set_ylabel("Held-out-surface prediction (N)")
    axes[1].set_ylabel("Held-out-surface prediction (N)")
    _add_object_key(key_ax, observations)
    fig.suptitle(
        "Held-out predictions: mass-only versus the best two-feature model",
        fontsize=13,
        y=0.95,
    )
    fig.text(
        0.06,
        0.045,
        "Each point was predicted without any condition from its physical surface in "
        "training or tuning; outputs are constrained to 0–8 N. "
        f"Excluded: {', '.join(excluded) if excluded else 'none'}.",
        ha="left",
        va="bottom",
        fontsize=7.8,
        color=MUTED,
    )
    _save_figure(fig, "matforce_regression_held_out_predictions")


def _predict_two_feature_model(model: Any, mass_g: Any, roughness: Any) -> np.ndarray:
    mass_values, roughness_values = np.broadcast_arrays(
        np.asarray(mass_g, dtype=float), np.asarray(roughness, dtype=float)
    )
    features = np.column_stack(
        (np.log10(mass_values.ravel()), roughness_values.ravel())
    )
    predictions = np.asarray(model.predict(features)).reshape(mass_values.shape)
    return np.clip(predictions, 0, 8)


def _make_response_surface_figure(
    observations: list[Observation], results: list[ModelResult], excluded: list[str]
) -> None:
    combined = _best_two_feature_result(results)
    model = combined.fitted_estimator
    masses = np.asarray([item.mass_g for item in observations])
    roughness = np.asarray([item.roughness_index for item in observations])
    forces = np.asarray([item.force_n for item in observations])

    mass_grid = np.geomspace(masses.min() * 0.75, masses.max() * 1.2, 180)
    roughness_padding = 0.03 * np.ptp(roughness)
    roughness_grid = np.linspace(
        max(0, roughness.min() - roughness_padding),
        roughness.max() + roughness_padding,
        180,
    )
    mass_mesh, roughness_mesh = np.meshgrid(mass_grid, roughness_grid)
    predicted_mesh = _predict_two_feature_model(model, mass_mesh, roughness_mesh)
    normalization = Normalize(vmin=0, vmax=8)

    fig = plt.figure(figsize=(13.2, 7.2), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.65, 1),
        left=0.07,
        right=0.94,
        top=0.89,
        bottom=0.12,
        hspace=0.34,
        wspace=0.28,
    )
    surface_ax = fig.add_subplot(grid[:, 0])
    mass_ax = fig.add_subplot(grid[0, 1])
    roughness_ax = fig.add_subplot(grid[1, 1])

    contour = surface_ax.contourf(
        mass_mesh,
        roughness_mesh,
        predicted_mesh,
        levels=np.linspace(0, 8, 17),
        cmap="viridis",
        norm=normalization,
        extend="max",
    )
    surface_ax.contour(
        mass_mesh,
        roughness_mesh,
        predicted_mesh,
        levels=np.arange(1, 8),
        colors="white",
        linewidths=0.55,
        alpha=0.65,
    )
    surface_ax.scatter(
        masses,
        roughness,
        c=forces,
        cmap="viridis",
        norm=normalization,
        s=80,
        marker="o",
        edgecolor="white",
        linewidth=1.1,
        zorder=3,
    )
    high_mass_threshold = float(np.quantile(masses, 0.9))
    for item in observations:
        is_high_mass = item.mass_g >= high_mass_threshold
        surface_ax.annotate(
            str(item.number),
            (item.mass_g, item.roughness_index),
            xytext=(-6 if is_high_mass else 4, 4),
            textcoords="offset points",
            ha="right" if is_high_mass else "left",
            color="white",
            fontsize=6.2,
            fontweight="bold",
            zorder=4,
        )
    surface_ax.set_xscale("log")
    surface_ax.set_xlabel("Object mass (g, log scale)")
    surface_ax.set_ylabel("Roughness index")
    surface_ax.set_title("Background = fitted force; circles = ground-truth force")
    surface_ax.grid(color="white", linewidth=0.5, alpha=0.35)
    colorbar = fig.colorbar(contour, ax=surface_ax, pad=0.02, fraction=0.05)
    colorbar.set_label("Minimum Gecko force (N)")

    roughness_quantiles = np.quantile(roughness, [0.25, 0.5, 0.75])
    curve_colors = ("#0b4f6c", BLUE, "#78a8d1")
    for value, color, label in zip(
        roughness_quantiles,
        curve_colors,
        ("Low roughness (Q1)", "Median roughness", "High roughness (Q3)"),
        strict=True,
    ):
        predictions = _predict_two_feature_model(
            model, mass_grid, np.full_like(mass_grid, value)
        )
        mass_ax.plot(mass_grid, predictions, color=color, linewidth=1.8, label=label)
    mass_ax.scatter(masses, forces, s=18, color=MUTED, alpha=0.35, zorder=1)
    mass_ax.set_xscale("log")
    mass_ax.set_xlabel("Mass (g, log scale)")
    mass_ax.set_ylabel("Predicted force (N)")
    mass_ax.set_ylim(-0.2, 8)
    mass_ax.set_title("Mass response at fixed roughness")
    _finish_axes(mass_ax)
    mass_ax.legend(frameon=False, fontsize=7.2, loc="upper left")

    mass_quantiles = np.quantile(masses, [0.25, 0.5, 0.75])
    for value, color, label in zip(
        mass_quantiles,
        curve_colors,
        ("Low mass (Q1)", "Median mass", "High mass (Q3)"),
        strict=True,
    ):
        predictions = _predict_two_feature_model(
            model, np.full_like(roughness_grid, value), roughness_grid
        )
        roughness_ax.plot(
            roughness_grid, predictions, color=color, linewidth=1.8, label=label
        )
    roughness_ax.scatter(roughness, forces, s=18, color=MUTED, alpha=0.35, zorder=1)
    roughness_ax.set_xlabel("Roughness index")
    roughness_ax.set_ylabel("Predicted force (N)")
    roughness_ax.set_ylim(-0.2, 8)
    roughness_ax.set_title("Roughness response at fixed mass")
    _finish_axes(roughness_ax)
    roughness_ax.legend(frameon=False, fontsize=7.2, loc="upper right")

    fig.suptitle(
        f"Combined mass-and-roughness regression: {combined.spec.label}",
        fontsize=13,
        y=0.965,
    )
    fig.text(
        0.07,
        0.025,
        "Surface is fitted on all labeled objects for interpretation only; predictions "
        "are constrained to 0–8 N and held-out accuracy is reported separately. "
        f"Full-data tuning: {_parameter_summary(model)}. "
        f"Excluded: {', '.join(excluded) if excluded else 'none'}.",
        ha="left",
        va="bottom",
        fontsize=7.5,
        color=MUTED,
    )
    _save_figure(fig, "matforce_two_feature_response_surface")


def _print_summary(
    observations: list[Observation], results: list[ModelResult], excluded: list[str]
) -> None:
    by_key = {result.spec.key: result for result in results}
    best_two_feature = _best_two_feature_result(results)
    targets = np.asarray([item.force_n for item in observations])
    log_masses = np.log10([item.mass_g for item in observations])
    roughness = np.asarray([item.roughness_index for item in observations])
    linear = by_key["two_feature_linear"].fitted_estimator
    scaler = linear.named_steps["scale"]
    regression = linear.named_steps["model"]
    original_coefficients = regression.coef_ / scaler.scale_
    original_intercept = regression.intercept_ - np.dot(
        original_coefficients, scaler.mean_
    )
    largest_errors = sorted(
        zip(
            observations,
            np.abs(best_two_feature.loo_predictions - targets),
            strict=True,
        ),
        key=lambda pair: pair[1],
        reverse=True,
    )[:5]

    surface_count = len({item.surface_id for item in observations})
    print(
        f"Evaluated {len(observations)} labeled conditions from {surface_count} surfaces; "
        f"excluded: {excluded or 'none'}"
    )
    for result in sorted(results, key=lambda item: item.loo_r2, reverse=True):
        print(
            f"{result.spec.label}: train R2={result.train_r2:.3f}, "
            f"grouped new-surface R2={result.loo_r2:.3f}, "
            f"RMSE={result.loo_rmse_n:.3f} N, MAE={result.loo_mae_n:.3f} N; "
            f"condition interpolation R2={result.interpolation_r2:.3f}; "
            f"n={result.sample_count}"
        )
    print(
        "Two-feature linear equation: force_N = "
        f"{original_intercept:.6f} + {original_coefficients[0]:.6f}*log10(mass_g) "
        f"+ {original_coefficients[1]:.9f}*roughness_index"
    )
    print(
        "Pearson correlations: "
        f"r(log10(mass), force)={np.corrcoef(log_masses, targets)[0, 1]:.3f}, "
        f"r(roughness, force)={np.corrcoef(roughness, targets)[0, 1]:.3f}, "
        f"r(log10(mass), roughness)={np.corrcoef(log_masses, roughness)[0, 1]:.3f}"
    )

    generator = np.random.default_rng(42)
    resamples = generator.integers(
        0, len(observations), size=(20_000, len(observations))
    )
    mass_errors = (
        targets[resamples] - by_key["mass_rbf"].loo_predictions[resamples]
    )
    combined_errors = (
        targets[resamples] - best_two_feature.loo_predictions[resamples]
    )
    rmse_difference = np.sqrt(np.mean(combined_errors**2, axis=1)) - np.sqrt(
        np.mean(mass_errors**2, axis=1)
    )
    lower, median, upper = np.quantile(rmse_difference, [0.025, 0.5, 0.975])
    print(
        "Paired bootstrap RMSE difference (best two-feature minus mass-only): "
        f"median={median:.3f} N, 95% interval=[{lower:.3f}, {upper:.3f}] N; "
        f"two-feature lower in {np.mean(rmse_difference < 0):.1%} of resamples"
    )
    print(
        f"Largest absolute {best_two_feature.spec.label} held-out errors: "
        + ", ".join(f"{item.name} ({error:.2f} N)" for item, error in largest_errors)
    )


def main() -> None:
    _set_paper_style()
    observations, excluded = _load_data()
    features = np.column_stack(
        (
            np.log10([item.mass_g for item in observations]),
            [item.roughness_index for item in observations],
            [
                item.projected_contact_fraction
                if item.projected_contact_fraction is not None
                else np.nan
                for item in observations
            ],
        )
    )
    targets = np.asarray([item.force_n for item in observations])
    surface_groups = np.asarray([item.surface_id for item in observations])
    config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    include_contact = bool(config.get("inputs", {}).get("use_projected_contact", False))
    results = _evaluate_models(
        features,
        targets,
        surface_groups,
        include_contact=include_contact,
    )
    _write_results(observations, results)
    _make_model_comparison_figure(results, excluded)
    _make_prediction_figure(observations, results, excluded)
    _make_response_surface_figure(observations, results, excluded)
    _print_summary(observations, results, excluded)


if __name__ == "__main__":
    main()
