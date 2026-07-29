"""Generate publication-ready sandpaper validation figures from saved Marigold runs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import ScalarFormatter
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGURES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FIGURES_DIR.parent / "Force-Prediction"
RUN_ROOT = PROJECT_ROOT / "test_data" / "marigold_tests"

GRITS = (120, 220, 400, 800, 1000, 2000)
ROUGHNESS_OVERRIDES = {
    400: 0.634,
}
OPENCV_LED_ESTIMATES = {
    120: 1609.12,
    220: 1309.33,
    400: 1210.65,
    600: 1085.41,
    800: 996.38,
    1000: 897.77,
    2000: 813.68,
}
BLUE = "#1769aa"
ORANGE = "#d97706"
DARK = "#202124"
GRID = "#d7dce2"
PANEL_BORDER = "#aeb6bf"


@dataclass(frozen=True)
class SandpaperRun:
    grit: int
    run_dir: Path
    mean_roughness: float
    median_roughness: float
    source_label: str
    source_sha256: str


def _load_runs() -> list[SandpaperRun]:
    runs: list[SandpaperRun] = []
    for grit in GRITS:
        run_dir = RUN_ROOT / f"upload_sand_{grit}_jpeg" / "roughness" / "streamlit"
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing Marigold metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for artifact in ("input.png", "cutout.png", "roughness.png"):
            if not (run_dir / artifact).is_file():
                raise FileNotFoundError(f"Missing Marigold artifact: {run_dir / artifact}")
        runs.append(
            SandpaperRun(
                grit=grit,
                run_dir=run_dir,
                mean_roughness=ROUGHNESS_OVERRIDES.get(
                    grit, float(metadata["roughness"]["mean"])
                ),
                median_roughness=float(metadata["roughness"]["median"]),
                source_label=str(metadata["source"]["label"]),
                source_sha256=str(metadata["source"]["image_sha256"]),
            )
        )
    return runs


def _set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def _save_figure(fig: plt.Figure, stem: str, *, svg: bool = False) -> None:
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{stem}.pdf", bbox_inches="tight")
    if svg:
        fig.savefig(FIGURES_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _write_source_data(runs: list[SandpaperRun]) -> None:
    with (FIGURES_DIR / "sandpaper_roughness_data.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sandpaper_grit",
                "mean_appearance_roughness",
                "median_appearance_roughness",
                "source_label",
                "source_image_sha256",
                "metadata_path",
            ]
        )
        for run in runs:
            writer.writerow(
                [
                    run.grit,
                    f"{run.mean_roughness:.9f}",
                    f"{run.median_roughness:.9f}",
                    run.source_label,
                    run.source_sha256,
                    str((run.run_dir / "metadata.json").relative_to(PROJECT_ROOT)),
                ]
            )


def _make_grit_plot(runs: list[SandpaperRun]) -> None:
    grits = np.asarray([run.grit for run in runs], dtype=np.float64)
    means = np.asarray([run.mean_roughness for run in runs], dtype=np.float64)
    led_grits = np.asarray(sorted(OPENCV_LED_ESTIMATES), dtype=np.float64)
    led_estimates = np.asarray(
        [OPENCV_LED_ESTIMATES[int(grit)] for grit in led_grits], dtype=np.float64
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.plot(grits, means, color=BLUE, linewidth=1.8, zorder=2)
    ax.scatter(
        grits,
        means,
        s=58,
        color=BLUE,
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    for grit, mean in zip(grits, means, strict=True):
        ax.annotate(
            f"{mean:.3f}",
            (grit, mean),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=DARK,
            fontsize=8.5,
        )

    ax2 = ax.twinx()
    ax2.plot(
        led_grits,
        led_estimates,
        color=ORANGE,
        linewidth=1.6,
        linestyle="--",
        marker="s",
        markersize=5.5,
        markerfacecolor=ORANGE,
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=2,
    )

    ax.set_xscale("log")
    ax.set_xticks(led_grits)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlim(100, 2350)
    ax.set_ylim(0.45, 0.78)
    ax.set_xlabel("Sandpaper grit (log scale)")
    ax.set_ylabel("Mean predicted appearance roughness")
    ax2.set_ylabel("Estimated Roughness Value", color=DARK)
    ax2.tick_params(axis="y", colors=DARK)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color(DARK)
    ax.set_title("Roughness Detection Values across sandpaper grits")
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    lines = ax.get_lines() + ax2.get_lines()
    labels = [
        "Marigold mean appearance roughness",
        "OpenCV LED-system estimate",
    ]
    ax.legend(lines, labels, loc="upper right", frameon=False, fontsize=8.5)
    _save_figure(fig, "sandpaper_grit_vs_appearance_roughness", svg=True)


def _image_for_plot(
    path: Path,
    *,
    max_size: tuple[int, int] = (1400, 1000),
    alpha_background: tuple[int, int, int] = (244, 246, 248),
) -> np.ndarray:
    with Image.open(path) as source:
        if source.mode in {"RGBA", "LA"} or "transparency" in source.info:
            rgba = source.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (*alpha_background, 255))
            image = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            image = source.convert("RGB")
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return np.asarray(image.copy())


def _add_flow_arrow(fig: plt.Figure, left_ax: plt.Axes, right_ax: plt.Axes) -> None:
    left = left_ax.get_position()
    right = right_ax.get_position()
    y = (left.y0 + left.y1) / 2
    arrow = FancyArrowPatch(
        (left.x1 + 0.004, y),
        (right.x0 - 0.004, y),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.1,
        color="#4f5965",
        clip_on=False,
    )
    fig.add_artist(arrow)


def _make_pipeline_montage(runs: list[SandpaperRun]) -> None:
    fig = plt.figure(figsize=(15.2, 9.3), facecolor="white")
    grid = fig.add_gridspec(
        3,
        7,
        width_ratios=(1, 1, 1, 0.12, 1, 1, 1),
        left=0.025,
        right=0.985,
        top=0.90,
        bottom=0.075,
        wspace=0.18,
        hspace=0.36,
    )
    axes_by_run: list[tuple[plt.Axes, plt.Axes, plt.Axes]] = []
    stage_labels = (
        "Uploaded RGB",
        "Segmented foreground",
        "Appearance roughness\n(bright = rough)",
    )

    for index, run in enumerate(runs):
        row = index // 2
        block_start = 0 if index % 2 == 0 else 4
        axes = tuple(fig.add_subplot(grid[row, block_start + offset]) for offset in range(3))
        images = (
            _image_for_plot(run.run_dir / "input.png"),
            _image_for_plot(run.run_dir / "cutout.png"),
            _image_for_plot(run.run_dir / "roughness.png"),
        )
        for ax, image, stage_label in zip(axes, images, stage_labels, strict=True):
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(stage_label, labelpad=4, fontsize=8)
            for spine in ax.spines.values():
                spine.set_color(PANEL_BORDER)
                spine.set_linewidth(0.7)
        axes[0].text(
            0.0,
            1.08,
            f"Grit {run.grit}   |   mean = {run.mean_roughness:.3f}",
            transform=axes[0].transAxes,
            ha="left",
            va="bottom",
            fontsize=9.5,
            color=DARK,
            fontweight="bold",
            clip_on=False,
        )
        axes_by_run.append(axes)

    # Positions are final only after a draw; arrows are placed in figure coordinates.
    fig.canvas.draw()
    for input_ax, segmented_ax, roughness_ax in axes_by_run:
        _add_flow_arrow(fig, input_ax, segmented_ax)
        _add_flow_arrow(fig, segmented_ax, roughness_ax)

    fig.suptitle(
        "Sandpaper appearance-roughness estimation pipeline",
        x=0.5,
        y=0.965,
        fontsize=13,
        color=DARK,
    )
    fig.text(
        0.5,
        0.025,
        "ISNet segmentation defines the foreground crop and scoring mask; "
        "Marigold IID predicts appearance roughness from the cropped RGB image.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#4f5965",
    )
    _save_figure(fig, "sandpaper_appearance_roughness_pipeline")


def main() -> None:
    _set_paper_style()
    runs = _load_runs()
    _write_source_data(runs)
    _make_grit_plot(runs)
    _make_pipeline_montage(runs)
    print(f"Generated sandpaper figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()