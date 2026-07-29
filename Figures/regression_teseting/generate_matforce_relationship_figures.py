"""Generate labeled MatForceFinal mass/roughness versus Gecko-force figures."""

from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import linregress, spearmanr

FIGURES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FIGURES_DIR.parents[1] / "Force-Prediction"
DATASET_PATH = PROJECT_ROOT / "data" / "MatForceFinal" / "dataset.csv"

# Keep Matplotlib's cache out of the source tree on read-only or sandboxed systems.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matforce-matplotlib-cache")
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

BLUE = "#1769aa"
DARK = "#202124"
MUTED = "#5f6368"
GRID = "#d7dce2"
FIT = "#6f7782"


@dataclass(frozen=True)
class MatForceObject:
    number: int
    name: str
    condition_id: str
    mass_g: float
    roughness_index: float
    gecko_force_n: float


def _optional_float(value: str | None) -> float | None:
    return float(value) if value and value.strip() else None


def _load_data() -> tuple[list[MatForceObject], list[str]]:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Missing MatForceFinal dataset: {DATASET_PATH}")

    with DATASET_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    required = {"Object", "Mass_g", "roughness_index", "gecko_force_n"}
    missing_columns = required.difference(rows[0] if rows else ())
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"MatForceFinal dataset is missing required columns: {names}")

    included: list[tuple[str, str, float, float, float]] = []
    excluded: list[str] = []
    for row in rows:
        name = row["Object"].strip()
        condition_id = (row.get("condition_id") or "baseline").strip() or "baseline"
        mass_g = _optional_float(row.get("Mass_g"))
        roughness = _optional_float(row.get("roughness_index"))
        force_n = _optional_float(row.get("gecko_force_n"))
        if mass_g is None or roughness is None or force_n is None:
            excluded.append(name)
            continue
        if mass_g <= 0:
            raise ValueError(f"{name}: mass must be positive, got {mass_g}")
        if roughness < 0:
            raise ValueError(f"{name}: roughness must be nonnegative, got {roughness}")
        if not 0 <= force_n <= 8:
            raise ValueError(f"{name}: Gecko force must be within 0–8 N, got {force_n}")
        display_name = name if condition_id == "baseline" else f"{name} ({condition_id})"
        included.append((display_name, condition_id, mass_g, roughness, force_n))

    included.sort(key=lambda item: item[0].casefold())
    objects = [
        MatForceObject(
            number=index,
            name=name,
            condition_id=condition_id,
            mass_g=mass_g,
            roughness_index=roughness,
            gecko_force_n=force_n,
        )
        for index, (name, condition_id, mass_g, roughness, force_n) in enumerate(
            included, start=1
        )
    ]
    if not objects:
        raise ValueError("MatForceFinal has no complete Gecko-force observations")
    return objects, excluded


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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def _save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _add_object_key(ax: plt.Axes, objects: list[MatForceObject]) -> None:
    ax.set_axis_off()
    ax.set_title("Object key", loc="left", pad=10, color=DARK)
    rows_per_column = 15
    x_positions = (0.0, 0.51)
    for index, item in enumerate(objects):
        column = index // rows_per_column
        row = index % rows_per_column
        x = x_positions[column]
        y = 0.965 - row * 0.064
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


def _scatter_numbered(
    ax: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    objects: list[MatForceObject],
) -> None:
    ax.scatter(
        x_values,
        y_values,
        s=62,
        color=BLUE,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    # Place compact numbered callouts in display coordinates. Crowded points are
    # handled first, and each label chooses the lowest-overlap candidate offset.
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
        Bbox.from_bounds(point[0] - 5, point[1] - 5, 10, 10)
        for point in display_points
    ]
    label_boxes: list[Bbox] = []
    axes_box = ax.get_window_extent()
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
        (23, 0),
        (-23, 0),
    )

    for index in order:
        x_value = x_values[index]
        y_value = y_values[index]
        item = objects[index]
        annotation = ax.annotate(
            str(item.number),
            xy=(x_value, y_value),
            xytext=candidate_offsets[0],
            textcoords="offset points",
            ha="center",
            va="center",
            color=BLUE,
            fontsize=6.5,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": BLUE,
                "linewidth": 0.55,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": BLUE,
                "linewidth": 0.45,
                "shrinkA": 2,
                "shrinkB": 4,
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


def _finish_axes(ax: plt.Axes) -> None:
    ax.set_ylim(-0.2, 8.15)
    ax.set_yticks(np.arange(0, 9, 1))
    ax.grid(color=GRID, linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def _add_statistics(
    ax: plt.Axes,
    *,
    count: int,
    pearson_r: float,
    spearman_rho: float,
    pearson_label: str,
) -> None:
    ax.text(
        0.02,
        0.98,
        f"n = {count}\n{pearson_label} = {pearson_r:.2f}\nSpearman ρ = {spearman_rho:.2f}",
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


def _make_mass_figure(objects: list[MatForceObject], excluded: list[str]) -> None:
    masses = np.asarray([item.mass_g for item in objects], dtype=np.float64)
    forces = np.asarray([item.gecko_force_n for item in objects], dtype=np.float64)
    log_masses = np.log10(masses)
    fit = linregress(log_masses, forces)
    rho = float(spearmanr(masses, forces).statistic)

    fig = plt.figure(figsize=(12.2, 6.6), facecolor="white")
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.65, 1),
        left=0.07,
        right=0.985,
        top=0.90,
        bottom=0.13,
        wspace=0.12,
    )
    ax = fig.add_subplot(grid[0, 0])
    key_ax = fig.add_subplot(grid[0, 1])

    x_fit = np.geomspace(masses.min() * 0.8, masses.max() * 1.2, 300)
    y_fit = fit.intercept + fit.slope * np.log10(x_fit)
    ax.plot(
        x_fit,
        y_fit,
        color=FIT,
        linewidth=1.4,
        linestyle="--",
        label="Least-squares fit vs. log₁₀(mass)",
        zorder=2,
    )
    _scatter_numbered(ax, masses, forces, objects)
    ax.set_xscale("log")
    ax.set_xlim(masses.min() * 0.65, masses.max() * 1.35)
    ax.set_xlabel("Object mass (g, log scale)")
    ax.set_ylabel("Minimum Gecko normal force (N)")
    ax.set_title("MatForceFinal: object mass versus Gecko force")
    _finish_axes(ax)
    _add_statistics(
        ax,
        count=len(objects),
        pearson_r=float(fit.rvalue),
        spearman_rho=rho,
        pearson_label="Pearson r (log mass)",
    )
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    _add_object_key(key_ax, objects)
    fig.text(
        0.07,
        0.035,
        f"Source: MatForceFinal/dataset.csv. Excluded incomplete force labels: "
        f"{', '.join(excluded) if excluded else 'none'}.",
        ha="left",
        va="bottom",
        fontsize=7.8,
        color=MUTED,
    )
    _save_figure(fig, "matforce_mass_vs_gecko_force")


def _make_roughness_figure(objects: list[MatForceObject], excluded: list[str]) -> None:
    roughness = np.asarray(
        [item.roughness_index for item in objects], dtype=np.float64
    )
    forces = np.asarray([item.gecko_force_n for item in objects], dtype=np.float64)
    fit = linregress(roughness, forces)
    rho = float(spearmanr(roughness, forces).statistic)

    fig = plt.figure(figsize=(12.2, 6.6), facecolor="white")
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.65, 1),
        left=0.07,
        right=0.985,
        top=0.90,
        bottom=0.13,
        wspace=0.12,
    )
    ax = fig.add_subplot(grid[0, 0])
    key_ax = fig.add_subplot(grid[0, 1])

    x_fit = np.linspace(roughness.min(), roughness.max(), 300)
    y_fit = fit.intercept + fit.slope * x_fit
    ax.plot(
        x_fit,
        y_fit,
        color=FIT,
        linewidth=1.4,
        linestyle="--",
        label="Least-squares linear fit",
        zorder=2,
    )
    _scatter_numbered(ax, roughness, forces, objects)
    padding = 0.06 * np.ptp(roughness)
    ax.set_xlim(roughness.min() - padding, roughness.max() + padding)
    ax.set_xlabel("Roughness index (unitless; higher = rougher)")
    ax.set_ylabel("Minimum Gecko normal force (N)")
    ax.set_title("MatForceFinal: roughness index versus Gecko force")
    _finish_axes(ax)
    _add_statistics(
        ax,
        count=len(objects),
        pearson_r=float(fit.rvalue),
        spearman_rho=rho,
        pearson_label="Pearson r",
    )
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    _add_object_key(key_ax, objects)
    fig.text(
        0.07,
        0.035,
        f"Source: MatForceFinal/dataset.csv. Excluded incomplete force labels: "
        f"{', '.join(excluded) if excluded else 'none'}.",
        ha="left",
        va="bottom",
        fontsize=7.8,
        color=MUTED,
    )
    _save_figure(fig, "matforce_roughness_vs_gecko_force")


def main() -> None:
    _set_paper_style()
    objects, excluded = _load_data()
    _make_mass_figure(objects, excluded)
    _make_roughness_figure(objects, excluded)
    print(
        f"Generated MatForceFinal relationship figures for {len(objects)} objects; "
        f"excluded: {', '.join(excluded) if excluded else 'none'}"
    )


if __name__ == "__main__":
    main()
