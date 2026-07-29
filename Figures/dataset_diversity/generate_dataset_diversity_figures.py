"""Generate publication-ready figures showing diversity in MatForceFinal."""

from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FIGURES_DIR = Path(__file__).resolve().parent
DATASET_PATH = (
    FIGURES_DIR.parents[1]
    / "Force-Prediction"
    / "data"
    / "MatForceFinal"
    / "dataset.csv"
)

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dataset-diversity-matplotlib-cache")
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402

BLUE = "#1769aa"
ORANGE = "#d97706"
DARK = "#202124"
MUTED = "#5f6368"
GRID = "#d7dce2"
SPLIT_COLORS = {
    "train": BLUE,
    "test": ORANGE,
}


@dataclass(frozen=True)
class DatasetObject:
    name: str
    split: str
    mass_g: float | None
    roughness_index: float | None


def _load_objects() -> list[DatasetObject]:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Missing dataset: {DATASET_PATH}")

    with DATASET_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    required = {"Object", "split", "Mass_g", "roughness_index"}
    missing = required.difference(rows[0] if rows else ())
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    objects: list[DatasetObject] = []
    for row in rows:
        name = row["Object"].strip()
        split = row["split"].strip().lower()
        mass = float(row["Mass_g"]) if row["Mass_g"].strip() else None
        roughness_index = (
            float(row["roughness_index"])
            if row["roughness_index"].strip()
            else None
        )
        if (
            not name
            or not split
            or (mass is not None and mass <= 0)
            or (roughness_index is not None and roughness_index < 0)
        ):
            raise ValueError(f"Invalid dataset row for {name!r}")
        objects.append(
            DatasetObject(
                name=name,
                split=split,
                mass_g=mass,
                roughness_index=roughness_index,
            )
        )

    if not objects:
        raise ValueError("Dataset contains no valid objects")
    return objects


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


def _split_order(objects: list[DatasetObject]) -> list[str]:
    present = {item.split for item in objects}
    preferred = [split for split in ("train", "test") if split in present]
    return preferred + sorted(present.difference(preferred))


def _split_color(split: str) -> str:
    return SPLIT_COLORS.get(split, MUTED)


def _split_legend(objects: list[DatasetObject]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=_split_color(split),
            markeredgecolor="white",
            markersize=6,
            label=split.title(),
        )
        for split in _split_order(objects)
    ]


def _make_mass_plot(objects: list[DatasetObject]) -> None:
    ordered = sorted(
        (item for item in objects if item.mass_g is not None),
        key=lambda item: item.mass_g,
    )
    if not ordered:
        raise ValueError("Dataset contains no measured masses")
    masses = np.asarray([item.mass_g for item in ordered], dtype=float)
    ranks = np.arange(1, len(ordered) + 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(masses, ranks, color=GRID, linewidth=1.2, zorder=1)
    for split in _split_order(ordered):
        mask = np.asarray([item.split == split for item in ordered])
        ax.scatter(
            masses[mask],
            ranks[mask],
            s=42,
            color=_split_color(split),
            edgecolor="white",
            linewidth=0.7,
            zorder=2,
        )

    ax.set_xscale("log")
    ax.set_xlim(2.5, 5000)
    ax.set_ylim(0, len(ordered) + 5)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_xlabel("Object mass (g, log scale)")
    ax.set_ylabel("Objects ordered from lightest to heaviest")
    ax.set_title("Measured mass diversity across the dataset")
    ax.text(
        0.02,
        0.96,
        f"n = {len(ordered)} objects  |  range = {masses.min():g}-{masses.max():g} g",
        transform=ax.transAxes,
        va="top",
        color=MUTED,
        fontsize=8.5,
    )
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=_split_legend(ordered),
        loc="upper left",
        bbox_to_anchor=(0.0, 0.89),
        frameon=False,
        title="Dataset split",
        fontsize=8,
        title_fontsize=8,
    )

    for index in (0, len(ordered) - 1):
        item = ordered[index]
        ax.annotate(
            f"{item.name}\n{item.mass_g:g} g",
            (item.mass_g, ranks[index]),
            xytext=(8, 0 if index == 0 else -5),
            textcoords="offset points",
            ha="left",
            va="bottom" if index == 0 else "top",
            color=DARK,
            fontsize=8,
        )
    _save_figure(fig, "dataset_mass_diversity")


def _make_roughness_plot(objects: list[DatasetObject]) -> None:
    ordered = sorted(
        (item for item in objects if item.roughness_index is not None),
        key=lambda item: item.roughness_index,
    )
    if not ordered:
        raise ValueError("Dataset contains no measured roughness indices")
    values = np.asarray([item.roughness_index for item in ordered], dtype=float)
    positions = np.arange(len(ordered))

    fig, ax = plt.subplots(figsize=(7.6, 8.0), constrained_layout=True)
    ax.hlines(
        positions,
        xmin=values.min() - 80,
        xmax=values,
        color=GRID,
        linewidth=0.8,
        zorder=1,
    )
    for split in _split_order(ordered):
        mask = np.asarray([item.split == split for item in ordered])
        ax.scatter(
            values[mask],
            positions[mask],
            s=48,
            color=_split_color(split),
            edgecolor="white",
            linewidth=0.7,
            zorder=2,
        )
    for value, position in zip(values, positions, strict=True):
        ax.text(
            value + 28,
            position,
            f"{value:.2f}",
            ha="left",
            va="center",
            color=DARK,
            fontsize=7.7,
        )

    ax.set_xlabel("High-frequency scattering index (a.u.)")
    ax.set_title("Measured optical roughness diversity across the dataset")
    ax.set_yticks(positions, [item.name for item in ordered])
    ax.set_xlim(values.min() - 100, values.max() + 260)
    ax.set_ylim(-0.8, len(ordered) - 0.2)
    ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(
        handles=_split_legend(ordered),
        loc="upper left",
        frameon=False,
        title="Dataset split",
        fontsize=8,
        title_fontsize=8,
    )
    ax.text(
        0.98,
        0.02,
        f"n = {len(ordered)} objects  |  range = {values.min():.2f}-{values.max():.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=MUTED,
        fontsize=8,
    )
    _save_figure(fig, "dataset_roughness_diversity")


def main() -> None:
    _set_paper_style()
    objects = _load_objects()
    _make_mass_plot(objects)
    _make_roughness_plot(objects)
    print(f"Generated dataset diversity figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
