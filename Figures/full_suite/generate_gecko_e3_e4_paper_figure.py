"""Generate the paper-ready Gecko E3/E4 grouped bar chart.

Run from the GSET workspace root:

    env/bin/python GSETGripper/Figures/full_suite/generate_gecko_e3_e4_paper_figure.py

The figure intentionally uses only three series:

* Ground-truth Gecko force.
* Displayed E3: source experiment E4 from the newest Gecko suite run available
  when this paper artifact was assembled (MAE 0.583 N).
* Displayed E4: source experiment E5 from the best saved Gecko run
  (MAE 0.205 N).

Both saved evaluations contain the same 15 objects and the same frozen truth
snapshot.  The checks below deliberately fail if that provenance or any plotted
arithmetic no longer agrees with the saved evaluation artifacts.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

# Keep Matplotlib's cache next to this script on sandboxed systems.
HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mpl"))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402
from matplotlib.ticker import MultipleLocator, ScalarFormatter  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402


FORCE_PREDICTION = HERE.parent.parent / "Force-Prediction"

# Exact, pinned artifacts make the paper figure reproducible.
LATEST_GECKO_SOURCE_E4 = FORCE_PREDICTION / (
    "data/MatForceFinal/results/evaluations/20260809T175747123610Z_e4/"
    "20260809T181141934097Z_2799cfe475b7.json"
)
BEST_GECKO_SOURCE_E5 = FORCE_PREDICTION / (
    "data/MatForceFinal/results/evaluations/20260809T024429516007Z_e5_curated/"
    "20260809T024429795360Z_2799cfe475b7.json"
)

OUTPUT_STEM = "gecko_e3_e4_force_by_object"
TRUTH_COLOR = "#26333D"
E3_COLOR = "#CC79A7"
E4_COLOR = "#D55E00"
GRID_COLOR = "#D8DEE5"
TEXT_COLOR = "#1F2933"
MUTED_COLOR = "#52606D"

# Correct the sole spelling error in the display copy without changing the
# underlying object ID or the audit CSV's source name.
DISPLAY_NAMES = {
    "avacado": "Avocado",
    "very_smooth_tile": "Smooth tile",
    "mold_release_spray_box": "Spray box",
    "large_hand_sanitizer": "Hand sanitizer",
    "headphones_case": "Headset case",
    "camera_cardboard_box": "Camera box",
}

# (square width as a fraction of source width, horizontal centre, vertical
# centre).  The default crop is already much tighter than the source photos;
# these overrides give the smallest objects extra magnification and keep the
# tall camera box fully visible.
THUMBNAIL_CROPS = {
    "very_smooth_tile": (0.38, 0.50, 0.69),
    "caprisun": (0.42, 0.50, 0.65),
    "small_book": (0.36, 0.50, 0.67),
    "avacado": (0.31, 0.47, 0.64),
    "rough_tile": (0.37, 0.50, 0.70),
    "clay_bowl": (0.43, 0.50, 0.68),
    "camera_cardboard_box": (0.79, 0.50, 0.61),
}


def _load_evaluation(path: Path, expected_experiment: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    evaluation = json.loads(path.read_text(encoding="utf-8"))
    metadata = evaluation["metadata"]
    if metadata.get("experiment") != expected_experiment:
        raise ValueError(
            f"{path.name} is {metadata.get('experiment')}, expected {expected_experiment}"
        )
    if metadata.get("active_grippers") != ["gecko"]:
        raise ValueError(f"{path.name} is not a Gecko-only evaluation")
    coverage = metadata.get("coverage", {})
    if coverage.get("evaluated") != 15 or coverage.get("skipped") != 0:
        raise ValueError(f"{path.name} does not contain 15 complete evaluations")
    return evaluation


def _saved_gecko_mae(evaluation: dict) -> float:
    return float(evaluation["metrics"]["force"]["gecko"]["mae"])


def _computed_mae(records: list[dict], prediction_key: str) -> float:
    return float(
        np.mean([abs(record[prediction_key] - record["truth"]) for record in records])
    )


def load_records() -> tuple[list[dict], dict]:
    """Load, join, and validate the two selected saved evaluations."""
    source_e4 = _load_evaluation(LATEST_GECKO_SOURCE_E4, "e4")
    source_e5 = _load_evaluation(BEST_GECKO_SOURCE_E5, "e5")

    e4_meta = source_e4["metadata"]
    e5_meta = source_e5["metadata"]
    truth_hash = e4_meta.get("truth_snapshot_sha256")
    if not truth_hash or truth_hash != e5_meta.get("truth_snapshot_sha256"):
        raise ValueError("selected E3 and E4 evaluations use different truth snapshots")

    e4_rows = {row["object_id"]: row for row in source_e4["rows"]}
    e5_rows = {row["object_id"]: row for row in source_e5["rows"]}
    if set(e4_rows) != set(e5_rows):
        raise ValueError("selected E3 and E4 evaluations contain different object IDs")

    records: list[dict] = []
    for object_id, e4_row in e4_rows.items():
        e5_row = e5_rows[object_id]
        e4_truth = float(e4_row["true_gecko_force_n"])
        e5_truth = float(e5_row["true_gecko_force_n"])
        if not np.isclose(e4_truth, e5_truth, rtol=0.0, atol=1e-12):
            raise ValueError(f"truth mismatch for {object_id}: {e4_truth} vs {e5_truth}")
        if e4_row.get("image_sha256") != e5_row.get("image_sha256"):
            raise ValueError(f"image mismatch for {object_id}")

        image_path = Path(e4_row["image_path"])
        image_path = (
            image_path if image_path.is_absolute() else FORCE_PREDICTION / image_path
        )
        records.append(
            {
                "object_id": object_id,
                "source_name": e4_row["object_name"],
                "display_name": DISPLAY_NAMES.get(object_id, e4_row["object_name"]),
                "image": image_path,
                "truth": e4_truth,
                # Source-E4 is displayed as paper experiment E3.
                "e3": float(e4_row["pred_gecko_force_n"]),
                # Source-E5 is displayed as paper experiment E4.
                "e4": float(e5_row["pred_gecko_force_n"]),
            }
        )

    e3_mae = _computed_mae(records, "e3")
    e4_mae = _computed_mae(records, "e4")
    if not np.isclose(e3_mae, _saved_gecko_mae(source_e4), atol=1e-12):
        raise ValueError("displayed E3 arithmetic does not match its saved evaluation")
    if not np.isclose(e4_mae, _saved_gecko_mae(source_e5), atol=1e-12):
        raise ValueError("displayed E4 arithmetic does not match its saved evaluation")
    if not np.isclose(e4_mae, 0.20486666666666672, atol=1e-12):
        raise ValueError("the pinned best E4 run is no longer the expected 0.205 N run")

    provenance = {
        "truth_snapshot_sha256": truth_hash,
        "displayed_e3_source_experiment": "e4",
        "displayed_e3_prediction_batch_id": e4_meta["prediction_batch_id"],
        "displayed_e3_evaluation_id": e4_meta["evaluation_id"],
        "displayed_e3_mae_n": e3_mae,
        "displayed_e4_source_experiment": "e5",
        "displayed_e4_prediction_batch_id": e5_meta["prediction_batch_id"],
        "displayed_e4_evaluation_id": e5_meta["evaluation_id"],
        "displayed_e4_mae_n": e4_mae,
    }
    return records, provenance


def _wrap_name(name: str, width: int = 6) -> str:
    words = str(name).replace("_", " ").split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def _object_thumbnail(
    path: Path, object_id: str, out_px: int = 420
) -> np.ndarray:
    """Return an aggressively cropped, square object thumbnail."""
    with Image.open(path) as opened:
        source = ImageOps.exif_transpose(opened).convert("RGB")

    width, height = source.size
    crop_fraction, center_x, center_y = THUMBNAIL_CROPS.get(
        object_id, (0.52, 0.50, 0.64)
    )
    side = min(width * crop_fraction, height * 0.94)
    left = min(max(width * center_x - side / 2.0, 0.0), width - side)
    top = min(max(height * center_y - side / 2.0, 0.0), height - side)
    cropped = source.crop((round(left), round(top), round(left + side), round(top + side)))
    resized = cropped.resize((out_px, out_px), Image.Resampling.LANCZOS)
    return np.asarray(resized)


def _add_object_strip(fig, ax, records: list[dict]) -> None:
    """Place large object crops and large names beneath their bar groups."""
    plot_box = ax.get_position()
    strip_ax = fig.add_axes(
        [plot_box.x0, 0.012, plot_box.width, plot_box.y0 - 0.037],
        frameon=False,
    )
    strip_ax.set_xlim(ax.get_xlim())
    strip_ax.set_ylim(0.0, 1.0)
    strip_ax.set_axis_off()
    transform = blended_transform_factory(strip_ax.transData, strip_ax.transAxes)

    for x, record in enumerate(records):
        image = _object_thumbnail(record["image"], record["object_id"])
        image_box = AnnotationBbox(
            OffsetImage(image, zoom=0.142),
            (x, 0.71),
            xycoords=transform,
            frameon=True,
            bboxprops={"edgecolor": "#C8D0D9", "linewidth": 1.0},
            pad=0.025,
            box_alignment=(0.5, 0.5),
            annotation_clip=False,
        )
        strip_ax.add_artist(image_box)
        # Alternate two compact label rows so large paper-scale type remains
        # readable even when neighboring object names are long.
        label_y = 0.48 if x % 2 == 0 else 0.28
        strip_ax.annotate(
            _wrap_name(record["display_name"]),
            (x, label_y),
            xycoords=transform,
            ha="center",
            va="top",
            fontsize=18,
            fontstretch="condensed",
            fontweight="normal",
            color=TEXT_COLOR,
            linespacing=1.05,
            annotation_clip=False,
        )


def _set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 20,
            "axes.titlesize": 28,
            "axes.titleweight": "bold",
            "axes.labelsize": 24,
            "axes.labelweight": "normal",
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": "#AEB8C2",
            "axes.linewidth": 1.1,
            "xtick.color": MUTED_COLOR,
            "ytick.color": MUTED_COLOR,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _write_audit_csv(records: list[dict], provenance: dict) -> Path:
    path = HERE / f"{OUTPUT_STEM}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "object_id",
            "source_object_name",
            "display_object_name",
            "ground_truth_gecko_force_n",
            "displayed_e3_predicted_force_n",
            "displayed_e3_absolute_error_n",
            "displayed_e4_predicted_force_n",
            "displayed_e4_absolute_error_n",
            "truth_snapshot_sha256",
            "displayed_e3_source_experiment",
            "displayed_e3_prediction_batch_id",
            "displayed_e3_evaluation_id",
            "displayed_e4_source_experiment",
            "displayed_e4_prediction_batch_id",
            "displayed_e4_evaluation_id",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "object_id": record["object_id"],
                    "source_object_name": record["source_name"],
                    "display_object_name": record["display_name"],
                    "ground_truth_gecko_force_n": f"{record['truth']:.6f}",
                    "displayed_e3_predicted_force_n": f"{record['e3']:.6f}",
                    "displayed_e3_absolute_error_n": f"{abs(record['e3'] - record['truth']):.6f}",
                    "displayed_e4_predicted_force_n": f"{record['e4']:.6f}",
                    "displayed_e4_absolute_error_n": f"{abs(record['e4'] - record['truth']):.6f}",
                    "truth_snapshot_sha256": provenance["truth_snapshot_sha256"],
                    "displayed_e3_source_experiment": provenance[
                        "displayed_e3_source_experiment"
                    ],
                    "displayed_e3_prediction_batch_id": provenance[
                        "displayed_e3_prediction_batch_id"
                    ],
                    "displayed_e3_evaluation_id": provenance[
                        "displayed_e3_evaluation_id"
                    ],
                    "displayed_e4_source_experiment": provenance[
                        "displayed_e4_source_experiment"
                    ],
                    "displayed_e4_prediction_batch_id": provenance[
                        "displayed_e4_prediction_batch_id"
                    ],
                    "displayed_e4_evaluation_id": provenance[
                        "displayed_e4_evaluation_id"
                    ],
                }
            )
    return path


def make_figure(records: list[dict], provenance: dict) -> None:
    ordered = sorted(records, key=lambda record: record["truth"])
    x = np.arange(len(ordered))
    truth = [record["truth"] for record in ordered]
    e3 = [record["e3"] for record in ordered]
    e4 = [record["e4"] for record in ordered]

    fig, ax = plt.subplots(figsize=(14.0, 5.6))
    width = 0.18
    ax.bar(
        x - width,
        truth,
        width,
        color=TRUTH_COLOR,
        label="Ground truth",
        zorder=3,
    )
    ax.bar(
        x,
        e3,
        width,
        color=E3_COLOR,
        label=f"E3  (MAE {provenance['displayed_e3_mae_n']:.3f} N)",
        zorder=3,
    )
    ax.bar(
        x + width,
        e4,
        width,
        color=E4_COLOR,
        label=f"E4  (MAE {provenance['displayed_e4_mae_n']:.3f} N)",
        zorder=3,
    )

    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.set_ylim(0.0, 7.05)
    ax.set_xlim(-0.50, len(ordered) - 0.50)
    ax.set_ylabel("Gecko force (N)", labelpad=8)
    ax.set_xticks(x, labels=[])
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", width=1.0, length=5)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.9, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Gecko force predictions by object", loc="left", pad=12)
    ax.legend(
        loc="upper left",
        ncol=3,
        frameon=False,
        columnspacing=1.25,
        handlelength=1.35,
        handletextpad=0.50,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.085, right=0.995, top=0.84, bottom=0.42)
    _add_object_strip(fig, ax, ordered)

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(HERE / f"{OUTPUT_STEM}.{suffix}", dpi=300)
    plt.close(fig)

    audit_path = _write_audit_csv(ordered, provenance)
    print(f"Truth snapshot: {provenance['truth_snapshot_sha256']}")
    print(
        "Displayed E3: "
        f"{provenance['displayed_e3_prediction_batch_id']} "
        f"(source E4, MAE {provenance['displayed_e3_mae_n']:.6f} N)"
    )
    print(
        "Displayed E4: "
        f"{provenance['displayed_e4_prediction_batch_id']} "
        f"(source E5, MAE {provenance['displayed_e4_mae_n']:.6f} N)"
    )
    print(f"Wrote {OUTPUT_STEM}.{{png,pdf,svg}}")
    print(f"Wrote {audit_path.name}")


def main() -> None:
    _set_paper_style()
    records, provenance = load_records()
    make_figure(records, provenance)


if __name__ == "__main__":
    main()
