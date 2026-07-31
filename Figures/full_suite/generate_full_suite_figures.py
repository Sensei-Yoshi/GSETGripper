"""Full-suite force-prediction figures for MatForceFinal (source E2 excluded).

Run from the GSET workspace root:

    env/bin/python GSETGripper/Figures/full_suite/generate_full_suite_figures.py

Data sources (all read straight from the saved benchmark artifacts):

  * Ground-truth Gecko force, object image, and name come from the MatForceFinal
    dataset loader.
  * Predicted Gecko force comes from the newest complete suite manifest under
    ``Force-Prediction/data/MatForceFinal/suites``. Reading the manifest keeps all
    plotted experiments on the same frozen split and suite run.

Experiment E2 is intentionally excluded from every figure.

It regenerates, into this folder:

  * ``suite_force_by_object.{png,pdf,svg}``  - truth and source E1/E3/E4/E5 bars.
  * ``suite_percentage_error.{png,pdf,svg}`` - source E1/E3/E4/E5 signed % error.
  * ``suite_e4_dumbbell.{png,pdf,svg}``       - source-E5 truth/prediction dumbbells
    and a signed relative-error table (displayed as E4 after relabeling).
  * ``suite_e4_percentage_errors.csv``       - exact source data for that table.

Source experiments E1/E3/E4/E5 are displayed consecutively as E1/E2/E3/E4
because source E2 is intentionally omitted from the comparison.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

# Keep Matplotlib's cache out of the source tree on sandboxed systems.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl"))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

HERE = Path(__file__).resolve().parent
FORCE_PREDICTION = HERE.parent.parent / "Force-Prediction"
SUITES_DIR = FORCE_PREDICTION / "data" / "MatForceFinal" / "suites"
sys.path.insert(0, str(FORCE_PREDICTION))

# Suite experiments in canonical order, deliberately WITHOUT E2.
EXPERIMENTS = ("e1", "e3", "e4", "e5")

# Colour-blind-safe palette matching the Streamlit reporting styles (E2 dropped).
EXPERIMENT_STYLES = {
    "e1": {"color": "#0072B2", "marker": "o", "label": "E1 vision-only"},
    "e3": {"color": "#009E73", "marker": "^", "label": "E2 semantic"},
    "e4": {"color": "#CC79A7", "marker": "D", "label": "E3 +mass"},
    "e5": {"color": "#D55E00", "marker": "v", "label": "E4 +roughness"},
}

TRUTH_COLOR = "#28323c"      # dark slate for ground truth
PRED_COLOR = "#e0891e"       # orange for the displayed E4 prediction series
GRID = "#d7dce2"
DARK = "#202124"
MUTED = "#5f6368"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _latest_complete_suite() -> tuple[Path, dict]:
    """Return the newest suite whose requested experiment runs all completed."""
    manifests = sorted(SUITES_DIR.glob("*/manifest.json"), reverse=True)
    for path in manifests:
        manifest = json.loads(path.read_text())
        runs = manifest.get("runs") or {}
        if all(
            runs.get(experiment, {}).get("status") == "completed"
            and runs[experiment].get("prediction_json_path")
            for experiment in EXPERIMENTS
        ):
            return path, manifest
    raise FileNotFoundError(
        "no complete suite contains source experiments E1, E3, E4, and E5"
    )


def _suite_batch_path(run: dict) -> Path:
    path = Path(run["prediction_json_path"])
    return path if path.is_absolute() else FORCE_PREDICTION / path


def load_records() -> list[dict]:
    """Assemble one record per test object with truth and per-experiment predictions."""
    from modules.config import load_config
    from modules.contracts import Gripper
    from modules.datasets import get_dataset

    cfg = load_config()
    cfg.dataset_id = "MatForceFinal"
    dataset = get_dataset(cfg, "MatForceFinal")

    manifest_path, manifest = _latest_complete_suite()
    suite_runs = manifest["runs"]
    split_hash = manifest.get("definition_snapshot", {}).get("split_sha256")

    predictions: dict[str, dict[str, float]] = {}
    batch_files: dict[str, str] = {}
    for experiment in EXPERIMENTS:
        path = _suite_batch_path(suite_runs[experiment])
        if not path.is_file():
            raise FileNotFoundError(
                f"suite {manifest['suite_id']} is missing prediction batch {path}"
            )
        batch_files[experiment] = path.name
        batch = json.loads(path.read_text())
        batch_split_hash = batch.get("metadata", {}).get("split_sha256")
        if split_hash and batch_split_hash != split_hash:
            raise ValueError(
                f"{path.name} split {batch_split_hash} does not match suite split {split_hash}"
            )
        rows = batch["rows"]
        predictions[experiment] = {
            row["object_id"]: row["pred_gecko_force_n"] for row in rows
        }

    ordered_ids = manifest.get("definition_snapshot", {}).get("split", {}).get("test")
    if not ordered_ids:
        ordered_ids = list(predictions["e5"])
    missing = {
        experiment: sorted(set(ordered_ids) - predictions[experiment].keys())
        for experiment in EXPERIMENTS
    }
    missing = {experiment: ids for experiment, ids in missing.items() if ids}
    if missing:
        raise ValueError(f"suite {manifest['suite_id']} has incomplete predictions: {missing}")

    records: list[dict] = []
    for object_id in ordered_ids:
        item = dataset.objects[object_id]
        outcome = item.gripper_outcomes.get(Gripper.GECKO)
        if outcome is None or outcome.min_force_n is None:
            continue
        image_path = item.image.path
        image_abs = image_path if os.path.isabs(image_path) else FORCE_PREDICTION / image_path
        records.append(
            {
                "object_id": object_id,
                "name": item.name,
                "true": float(outcome.min_force_n),
                "image": Path(image_abs),
                "preds": {
                    experiment: predictions[experiment][object_id]
                    for experiment in EXPERIMENTS
                },
            }
        )
    print(f"Suite used: {manifest['suite_id']}")
    print(f"Manifest: {manifest_path}")
    print(f"Split SHA-256: {split_hash}")
    print("Prediction batches used:")
    for experiment in EXPERIMENTS:
        print(f"  {experiment.upper()}: {batch_files[experiment]}")
    return records


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _set_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 12,
            "axes.edgecolor": "#c2c9d2",
            "axes.linewidth": 1.0,
            "axes.titlesize": 13,
            "axes.labelcolor": DARK,
            "text.color": DARK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )


def _zoomed_thumbnail(path: Path, out_px: int = 240) -> np.ndarray | None:
    """Return a square, zoomed-in RGB thumbnail that fills the frame with the object.

    The raw photos are centred table-top shots with wide side margins, so we crop
    the central column and remove the upper 25% of the full image height before
    resizing. That removes background and makes the object read clearly at
    thumbnail scale.
    """
    try:
        with Image.open(path) as opened:
            source = ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, ValueError):
        return None
    width, height = source.size
    crop_w = width * 0.60          # keep the central 60% width (drop side margins)
    crop_h = height * 0.75         # requested 25% reduction in the full image height
    left = (width - crop_w) / 2.0
    top = height - crop_h          # discard the mostly empty upper quarter
    cropped = source.crop((int(left), int(top), int(left + crop_w), int(top + crop_h)))
    # Letterbox into a white square, then upscale for a crisp thumbnail.
    side = max(cropped.size)
    square = Image.new("RGB", (side, side), "white")
    square.paste(cropped, ((side - cropped.width) // 2, (side - cropped.height) // 2))
    square = square.resize((out_px, out_px), Image.LANCZOS)
    return np.asarray(square)


def _wrap_name(name: str, width: int = 12) -> str:
    words = str(name).replace("_", " ").split()
    lines: list[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 <= width:
            current = f"{current} {word}".strip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


def _add_thumb_strip(
    fig,
    ax,
    records: list[dict],
    *,
    zoom: float = 0.36,
) -> None:
    """Draw thumbnails and names in separate rows below the plotting axes."""
    plot_box = ax.get_position()
    strip_bottom = 0.012
    strip_top = plot_box.y0 - 0.025
    strip_ax = fig.add_axes(
        [plot_box.x0, strip_bottom, plot_box.width, strip_top - strip_bottom],
        frameon=False,
    )
    strip_ax.set_xlim(ax.get_xlim())
    strip_ax.set_ylim(0, 1)
    strip_ax.set_axis_off()
    trans = blended_transform_factory(strip_ax.transData, strip_ax.transAxes)
    for x, record in enumerate(records):
        image = _zoomed_thumbnail(record["image"])
        if image is not None:
            box = AnnotationBbox(
                OffsetImage(image, zoom=zoom),
                (x, 0.68),
                xycoords=trans,
                frameon=True,
                bboxprops={"edgecolor": "#d1d5db", "linewidth": 0.7},
                pad=0.03,
                box_alignment=(0.5, 0.5),
                annotation_clip=False,
            )
            strip_ax.add_artist(box)
        strip_ax.annotate(
            _wrap_name(record["name"]),
            (x, 0.03),
            xycoords=trans,
            ha="center", va="bottom", fontsize=8.5, color=DARK,
            annotation_clip=False,
        )


def _use_plain_y_ticks(ax) -> None:
    """Keep numeric axes in fixed-point notation with no scientific offset."""
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)


def _save(fig, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(HERE / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.{{png,pdf,svg}}")


def _mae(records: list[dict], experiment: str) -> tuple[float, int]:
    values = [
        abs(r["preds"][experiment] - r["true"])
        for r in records
        if r["preds"].get(experiment) is not None
    ]
    return (float(np.mean(values)) if values else float("nan"), len(values))


def _signed_percent_error(prediction: float, truth: float) -> float:
    """Return signed percentage error, rejecting undefined zero-truth cases."""
    if truth == 0:
        raise ValueError("percentage error is undefined when ground truth is zero")
    return 100.0 * (prediction - truth) / truth


def _write_e4_error_table(records: list[dict], errors: list[float]) -> None:
    """Write the exact values used by the rendered E4 percentage-error table."""
    path = HERE / "suite_e4_percentage_errors.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Object name", "Percent error (%)"])
        for record, error in zip(records, errors, strict=True):
            writer.writerow([record["name"], f"{error:.6f}"])
    print(f"  wrote {path.name}")


# --------------------------------------------------------------------------- #
# Suite grouped bars (source E1/E3/E4/E5, displayed as E1/E2/E3/E4)
# --------------------------------------------------------------------------- #
def make_suite_force_by_object(records: list[dict]) -> None:
    ordered = sorted(records, key=lambda r: r["true"])
    x = np.arange(len(ordered))
    truth = [r["true"] for r in ordered]

    fig, ax = plt.subplots(figsize=(15.5, 8.2), facecolor="white")
    width = 0.16
    series_count = len(EXPERIMENTS) + 1
    offsets = (np.arange(series_count) - (series_count - 1) / 2) * width
    ax.bar(
        x + offsets[0],
        truth,
        width,
        color=TRUTH_COLOR,
        zorder=3,
        label="Ground truth",
    )
    plotted_values = list(truth)
    for offset, experiment in zip(offsets[1:], EXPERIMENTS, strict=True):
        style = EXPERIMENT_STYLES[experiment]
        series = [r["preds"][experiment] for r in ordered]
        plotted_values.extend(series)
        mae, _ = _mae(ordered, experiment)
        ax.bar(
            x + offset,
            series,
            width,
            color=style["color"],
            zorder=3,
            label=f"{style['label']}  (MAE {mae:.2f} N)",
        )

    ax.set_ylim(0, max(max(truth), max(plotted_values)) * 1.15)
    _use_plain_y_ticks(ax)
    ax.set_ylabel("Gecko force (N)")
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks(x, labels=[])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.7, len(ordered) - 0.3)
    ax.set_title("Force predictions by object", loc="left", pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=10, ncol=3)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.29)
    _add_thumb_strip(fig, ax, ordered)
    _save(fig, "suite_force_by_object")


def make_suite_percentage_error(records: list[dict]) -> None:
    ordered = sorted(records, key=lambda r: r["true"])
    x = np.arange(len(ordered))

    fig, ax = plt.subplots(figsize=(15.5, 8.2), facecolor="white")
    ax.axhline(0, color=DARK, linewidth=1.0, linestyle="--", zorder=1, label="Zero error")
    for experiment in EXPERIMENTS:
        style = EXPERIMENT_STYLES[experiment]
        series = [
            _signed_percent_error(r["preds"][experiment], r["true"])
            for r in ordered
        ]
        ax.plot(x, series, color=style["color"], linewidth=1.5, alpha=0.9,
                marker=style["marker"], markersize=8, markeredgecolor="white",
                markeredgewidth=0.6, zorder=3, label=style["label"])

    ax.set_ylabel("Signed percentage error (%)")
    _use_plain_y_ticks(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks(x, labels=[])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.7, len(ordered) - 0.3)
    ax.set_title("Signed percentage error by object", loc="left", pad=12)
    ax.legend(loc="upper right", frameon=False, fontsize=10, ncol=3)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.29)
    _add_thumb_strip(fig, ax, ordered)
    _save(fig, "suite_percentage_error")


def make_suite_e4_dumbbell(records: list[dict]) -> None:
    """Compare displayed E4 (source E5) with truth and tabulate relative error."""
    ordered = sorted(records, key=lambda r: r["true"], reverse=True)
    x = np.arange(len(ordered))
    truth = [r["true"] for r in ordered]
    pred = [r["preds"]["e5"] for r in ordered]
    rel_err = [_signed_percent_error(p, t) for p, t in zip(pred, truth, strict=True)]

    fig, (ax, ax_table) = plt.subplots(
        2,
        1,
        figsize=(15.5, 11.8),
        height_ratios=[2.2, 1.8],
        sharex=True,
        facecolor="white",
    )

    ax.vlines(
        x,
        np.minimum(truth, pred),
        np.maximum(truth, pred),
        color="#9aa5b1",
        linewidth=2.0,
        zorder=1,
    )
    ax.scatter(
        x,
        truth,
        s=95,
        color=TRUTH_COLOR,
        edgecolors="white",
        linewidths=1.1,
        zorder=3,
        label="Ground truth",
    )
    ax.scatter(
        x,
        pred,
        s=95,
        color=PRED_COLOR,
        edgecolors="white",
        linewidths=1.1,
        zorder=3,
        label="E4 predicted (+roughness)",
    )
    ax.set_ylim(0, max(max(truth), max(pred)) * 1.15)
    _use_plain_y_ticks(ax)
    ax.set_ylabel("Gecko force (N)")
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("E4 predicted vs ground-truth Gecko force", loc="left", pad=12)
    ax.legend(loc="upper right", frameon=False, fontsize=11)

    mae, count = _mae(ordered, "e5")
    ax.text(
        0.015,
        0.045,
        f"MAE = {mae:.3f} N (n={count})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        color=MUTED,
    )

    table_rows = [
        [record["name"], f"{error:+.2f}%"]
        for record, error in zip(ordered, rel_err, strict=True)
    ]
    table = ax_table.table(
        cellText=table_rows,
        colLabels=["Object name", "Percent error"],
        cellLoc="left",
        colLoc="left",
        colWidths=[0.76, 0.24],
        bbox=[0.12, 0.02, 0.76, 0.94],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor(TRUTH_COLOR)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f8fafc")
        else:
            cell.set_facecolor("white")
        if column == 1:
            cell.get_text().set_ha("right")
    ax_table.set_title("Signed percentage error by object", loc="left", pad=6)
    ax_table.set_xlim(-0.7, len(ordered) - 0.3)
    ax_table.set_axis_off()

    _write_e4_error_table(ordered, rel_err)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.95, bottom=0.22, hspace=0.16)
    _add_thumb_strip(fig, ax_table, ordered)
    _save(fig, "suite_e4_dumbbell")


def main() -> None:
    _set_paper_style()
    records = load_records()
    print(f"Loaded {len(records)} test objects.\nGenerating figures:")
    make_suite_force_by_object(records)
    make_suite_percentage_error(records)
    make_suite_e4_dumbbell(records)
    print("Done.")


if __name__ == "__main__":
    main()
