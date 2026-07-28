"""Independent Marigold appearance-roughness and topography workbench."""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from modules.models.background_remover import (
    DEFAULT_BACKGROUND_MODEL,
    BackgroundRemover,
)
from modules.models.marigold_rough import (
    DEFAULT_CONTACT_BAND_FRACTION,
    DEFAULT_CROP_PADDING_RATIO,
    DEFAULT_ENSEMBLE_SIZE,
    DEFAULT_INFERENCE_STEPS,
    DEFAULT_MASK_EROSION_RATIO,
    DEFAULT_PROCESSING_RESOLUTION,
    MarigoldAnalyzer,
    available_device,
    list_saved_runs,
    run_marigold,
)
from modules.models.marigold_topo import (
    DEFAULT_BASE_SURFACE_SIGMA_RATIO,
    MarigoldNormalsAnalyzer,
    list_saved_topography_runs,
    run_marigold_topography,
)
from streamlit_app.context import AppContext

ROUGHNESS = "Roughness"
TOPOGRAPHY = "Topography"
STABLE_RUN_KEY = "streamlit"


@dataclass(frozen=True)
class _SelectedImage:
    display_label: str
    image: Image.Image
    source_label: str
    dataset_id: str | None
    object_id: str | None
    source_path: str | None
    roughness_root: Path
    topography_root: Path
    target_key: str


@st.cache_resource(show_spinner=False)
def _roughness_analyzer(device: str, processing_resolution: int) -> MarigoldAnalyzer:
    return MarigoldAnalyzer(device=device, processing_resolution=processing_resolution)


@st.cache_resource(show_spinner=False)
def _topography_analyzer(device: str, processing_resolution: int) -> MarigoldNormalsAnalyzer:
    return MarigoldNormalsAnalyzer(device=device, processing_resolution=processing_resolution)


@st.cache_resource(show_spinner=False)
def _background_remover(model_name: str) -> BackgroundRemover:
    return BackgroundRemover(model_name=model_name)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "image"


def _open_upload(uploaded: Any) -> Image.Image | None:
    if uploaded is None:
        return None
    try:
        with Image.open(io.BytesIO(uploaded.getvalue())) as source:
            return ImageOps.exif_transpose(source).convert("RGB")
    except (OSError, UnidentifiedImageError):
        return None


def _manual_region_controls(selection: _SelectedImage) -> tuple[int, int, int, int]:
    """Render normalized ROI controls and return an original-image XYXY box."""
    key = _slug(selection.target_key)
    horizontal = st.slider(
        "Horizontal extent (%)",
        0,
        100,
        (15, 85),
        key=f"marigold_manual_horizontal_{key}",
    )
    vertical = st.slider(
        "Vertical extent (%)",
        0,
        100,
        (15, 85),
        key=f"marigold_manual_vertical_{key}",
    )
    width, height = selection.image.size
    x0 = min(width - 1, round(horizontal[0] * width / 100))
    x1 = max(x0 + 1, min(width, round(horizontal[1] * width / 100)))
    y0 = min(height - 1, round(vertical[0] * height / 100))
    y1 = max(y0 + 1, min(height, round(vertical[1] * height / 100)))

    preview = selection.image.copy()
    line_width = max(2, round(min(width, height) / 150))
    ImageDraw.Draw(preview).rectangle(
        (x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)),
        outline=(255, 45, 45),
        width=line_width,
    )
    st.image(preview, caption=f"Selected region: {x0}, {y0} to {x1}, {y1}", width="stretch")
    return x0, y0, x1, y1


def _prepare_manual_region(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    padding_ratio: float,
) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
    """Crop around a manual ROI and build its scoring mask in crop coordinates."""
    x0, y0, x1, y1 = bbox
    width, height = image.size
    pad_x = round((x1 - x0) * padding_ratio)
    pad_y = round((y1 - y0) * padding_ratio)
    px0, py0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
    px1, py1 = min(width, x1 + pad_x), min(height, y1 + pad_y)
    crop = image.crop((px0, py0, px1, py1))
    mask = Image.new("L", crop.size, color=0)
    ImageDraw.Draw(mask).rectangle(
        (x0 - px0, y0 - py0, x1 - px0 - 1, y1 - py0 - 1),
        fill=255,
    )
    return crop, mask, (px0, py0, px1, py1)


def _artifact_path(run: dict[str, Any], name: str) -> Path | None:
    artifact = run.get("artifacts", {}).get(name)
    if not artifact:
        return None
    path = Path(str(run["run_dir"])) / str(artifact)
    return path if path.is_file() else None


def _run_caption(run: dict[str, Any]) -> None:
    source = run.get("source", {})
    model = run.get("model", {})
    st.caption(
        f"{source.get('label') or run.get('run_id', 'Marigold run')} · "
        f"{run.get('created_at', 'Unknown time')} · {model.get('id', 'Unknown model')} · "
        f"{model.get('device', 'unknown device')} · {run.get('duration_seconds', 0):g} s"
    )


def _render_artifact_row(
    run: dict[str, Any],
    items: tuple[tuple[str, str], ...],
) -> None:
    columns = st.columns(len(items), gap="medium")
    for column, (artifact, caption) in zip(columns, items, strict=True):
        path = _artifact_path(run, artifact)
        with column:
            if path:
                st.image(str(path), caption=caption, width="stretch")
            else:
                st.caption(f"{caption} is unavailable for this run.")


def _render_quality(run: dict[str, Any]) -> None:
    quality = run.get("quality") or {}
    if quality.get("status") == "warning":
        st.warning("Quality checks: " + ", ".join(quality.get("warnings", [])))
    elif quality.get("status") == "ok":
        st.success("Mask and scoring-region quality checks passed.")


def _render_provenance(run: dict[str, Any], analysis: str) -> None:
    with st.expander(f"{analysis} data and provenance"):
        display = {key: value for key, value in run.items() if key != "run_dir"}
        st.json(display, expanded=True)
        st.caption(f"Saved artifacts: {run['run_dir']}")
        st.download_button(
            "Download metadata JSON",
            data=json.dumps(display, indent=2).encode(),
            file_name=f"{analysis.lower()}_{run.get('run_id', 'marigold')}_metadata.json",
            mime="application/json",
            key=(
                f"marigold_{analysis.lower()}_download_"
                f"{_slug(str(run.get('source', {}).get('object_id') or run.get('source', {}).get('label') or run.get('run_dir', 'image')))}_"
                f"{run.get('run_id', 'unknown')}"
            ),
        )


def _render_roughness_run(run: dict[str, Any]) -> None:
    st.subheader("Appearance roughness")
    _run_caption(run)
    _render_artifact_row(
        run,
        (
            ("input", "Input"),
            ("foreground_preview", "Foreground preview"),
            ("cutout", "Transparent cutout"),
            ("foreground_mask", "Source foreground mask"),
        ),
    )
    _render_artifact_row(
        run,
        (
            ("inference_crop", "Marigold input crop"),
            ("analysis_foreground_mask", "Analysis foreground"),
            ("scoring_mask", "Scored grasp band"),
        ),
    )
    _render_artifact_row(
        run,
        (
            ("albedo", "Albedo"),
            ("roughness", "Appearance roughness (bright = rough)"),
            ("metallicity", "Metallicity (bright = metallic)"),
        ),
    )
    uncertainty_path = _artifact_path(run, "roughness_uncertainty")
    if uncertainty_path:
        st.image(
            str(uncertainty_path),
            caption="Roughness uncertainty (bright = less certain)",
            width="stretch",
        )

    roughness = run.get("roughness", {})
    metallicity = run.get("metallicity", {})
    uncertainty = run.get("roughness_uncertainty") or {}
    metrics = st.columns(5)
    metrics[0].metric("Mean roughness", f"{roughness.get('mean', float('nan')):.3f}")
    metrics[1].metric("Median roughness", f"{roughness.get('median', float('nan')):.3f}")
    metrics[2].metric(
        "Roughness IQR",
        f"{roughness.get('p75', float('nan')) - roughness.get('p25', float('nan')):.3f}",
    )
    metrics[3].metric("Mean metallicity", f"{metallicity.get('mean', float('nan')):.3f}")
    metrics[4].metric(
        "Mean uncertainty",
        f"{uncertainty['mean']:.3f}" if uncertainty.get("mean") is not None else "Unavailable",
    )
    _render_quality(run)
    st.caption(
        "This is the IID model's BRDF appearance roughness: how broadly the material "
        "reflects light. It is not physical bump height, friction, or profilometer Ra."
    )
    _render_provenance(run, "Roughness")


def _render_topography_run(run: dict[str, Any]) -> None:
    st.subheader("Surface topography")
    _run_caption(run)
    if _artifact_path(run, "input"):
        _render_artifact_row(
            run,
            (
                ("input", "Input"),
                ("foreground_preview", "Foreground preview"),
                ("cutout", "Transparent cutout"),
                ("source_foreground_mask", "Source foreground mask"),
            ),
        )
    _render_artifact_row(
        run,
        (
            ("input_crop", "Marigold input crop"),
            ("foreground_mask", "Analysis foreground"),
            ("scoring_mask", "Scored grasp band"),
        ),
    )
    _render_artifact_row(
        run,
        (
            ("normal_map", "Predicted local surface normals"),
            ("base_normal_map", "Smoothed base-shape normals"),
            ("bump_angle", "Local angular residual (bright = stronger relief)"),
        ),
    )
    uncertainty_path = _artifact_path(run, "normal_uncertainty")
    if uncertainty_path:
        st.image(
            str(uncertainty_path),
            caption="Normal uncertainty (bright = less certain)",
            width="stretch",
        )

    topography = run.get("topographic_roughness", {})
    angles = topography.get("angle_degrees", {})
    uncertainty = run.get("normal_uncertainty") or {}
    metrics = st.columns(6)
    metrics[0].metric("Topography score", f"{topography.get('score_0_1', float('nan')):.3f}")
    metrics[1].metric("Mean residual", f"{angles.get('mean', float('nan')):.2f}°")
    metrics[2].metric("Median residual", f"{angles.get('median', float('nan')):.2f}°")
    metrics[3].metric("P75 residual", f"{angles.get('p75', float('nan')):.2f}°")
    metrics[4].metric("P95 residual", f"{angles.get('p95', float('nan')):.2f}°")
    metrics[5].metric(
        "Mean uncertainty",
        f"{uncertainty['mean']:.3f}" if uncertainty.get("mean") is not None else "Unavailable",
    )
    _render_quality(run)
    st.caption(
        "The normal map estimates local orientation. The base-normal map is that field "
        "heavily smoothed to represent the object's broad curvature. The residual image and "
        "score measure their angular difference inside the grasp band, so raised bumps and "
        "grooves remain while a can's overall cylindrical curve is largely removed."
    )
    _render_provenance(run, "Topography")


def _dataset_selection(context: AppContext, row: Any) -> _SelectedImage | None:
    image_path = context.config.root / row.image.path
    if not image_path.is_file():
        st.warning(f"The dataset image for {row.name!r} is not available locally.")
        return None
    try:
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (OSError, UnidentifiedImageError):
        st.warning(f"The dataset image for {row.name!r} could not be decoded.")
        return None
    base = context.dataset.paths.object_dir(row.object_id)
    return _SelectedImage(
        display_label=f"{row.name} ({row.object_id})",
        image=image,
        source_label=row.name,
        dataset_id=context.dataset.dataset_id,
        object_id=row.object_id,
        source_path=str(row.image.path),
        roughness_root=base / "roughness",
        topography_root=base / "topography",
        target_key=str(base),
    )


def _upload_selection(context: AppContext, uploaded: Any) -> _SelectedImage | None:
    image = _open_upload(uploaded)
    if image is None:
        st.error(f"The uploaded file {uploaded.name!r} could not be decoded as an image.")
        return None
    base = (
        context.config.root
        / "test_data"
        / "marigold_tests"
        / f"upload_{_slug(str(uploaded.name))}"
    )
    return _SelectedImage(
        display_label=f"Upload: {uploaded.name}",
        image=image,
        source_label=str(uploaded.name),
        dataset_id=None,
        object_id=None,
        source_path=None,
        roughness_root=base / "roughness",
        topography_root=base / "topography",
        target_key=str(base),
    )


def _history_section(
    analyses: list[str],
    selections: list[_SelectedImage],
) -> None:
    if not selections:
        st.info("Select at least one dataset or uploaded image to view its saved results.")
        return
    for image_index, selection in enumerate(selections):
        if len(selections) > 1:
            st.subheader(selection.display_label)
        for analysis in analyses:
            if analysis == ROUGHNESS:
                runs = list_saved_runs(selection.roughness_root)
                renderer = _render_roughness_run
            else:
                runs = list_saved_topography_runs(selection.topography_root)
                renderer = _render_topography_run
            if not runs:
                st.info(
                    f"No saved {analysis.lower()} result exists for "
                    f"{selection.display_label}."
                )
                continue
            labels = {
                (
                    f"{run.get('created_at', '')[:19]} | "
                    f"{run.get('source', {}).get('label', run.get('run_id', 'run'))} | "
                    f"{run.get('run_id', 'run')}"
                ): run
                for run in runs
            }
            selected = st.selectbox(
                f"Saved {analysis.lower()} result for {selection.display_label}",
                list(labels),
                key=(
                    f"marigold_history_{analysis.lower()}_{image_index}_"
                    f"{_slug(selection.target_key)}"
                ),
            )
            renderer(labels[selected])


def _run_workbench(context: AppContext) -> None:
    rows = sorted(context.rows, key=lambda row: (row.name.casefold(), row.object_id))
    labels = {f"{row.name} ({row.object_id})": row for row in rows}
    selected_labels: list[str] = []
    if labels:
        selected_labels = st.multiselect(
            "Marigold dataset images",
            list(labels),
            default=[next(iter(labels))],
            key="roughness_dataset_images",
            help="Every selected object is processed with the same analysis settings.",
        )
    else:
        st.info("This dataset contains no indexed images. Upload images below.")

    uploaded_files = st.file_uploader(
        "Upload one or more additional images",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        key="roughness_override_images",
        help=(
            "Choose multiple files in the picker with Shift-click or Command-click. "
            "You can also reopen the picker to add more files."
        ),
    )
    st.caption(
        "The uploader accepts multiple files. On macOS, use Shift-click for a range or "
        "Command-click for individual files."
    )
    selections = [
        selection
        for selection in (
            _dataset_selection(context, labels[label]) for label in selected_labels
        )
        if selection is not None
    ]
    selections.extend(
        selection
        for selection in (
            _upload_selection(context, uploaded) for uploaded in (uploaded_files or [])
        )
        if selection is not None
    )

    if selections:
        st.caption(f"{len(selections)} image{'s' if len(selections) != 1 else ''} selected")
        preview_columns = st.columns(min(4, len(selections)), gap="medium")
        for index, selection in enumerate(selections):
            with preview_columns[index % len(preview_columns)]:
                st.image(selection.image, caption=selection.display_label, width="stretch")

    analyses = st.multiselect(
        "Marigold analyses",
        [ROUGHNESS, TOPOGRAPHY],
        default=[ROUGHNESS, TOPOGRAPHY],
        key="marigold_analysis_modes",
        help=(
            "Roughness uses the IID appearance checkpoint. Topography uses the separate "
            "normals checkpoint and measures local relief after removing broad curvature."
        ),
    )
    if not analyses:
        st.warning("Select at least one analysis.")

    manual_region = st.checkbox(
        "Manually select the analysis region (override automatic segmentation)",
        value=False,
        key="marigold_manual_region",
        help=(
            "Skips background removal, crops each image around your rectangle, and scores "
            "only pixels inside that rectangle."
        ),
    )

    view_history = st.toggle(
        "View saved Marigold results",
        value=False,
        key="roughness_view_history",
    )
    if view_history:
        _history_section(analyses, selections)
        return

    with st.expander("Marigold settings"):
        processing_resolution = st.select_slider(
            "Processing resolution",
            options=[384, 512, 640, 768],
            value=DEFAULT_PROCESSING_RESOLUTION,
            key="roughness_processing_resolution",
            help="Both Marigold v1.1 checkpoints are optimized around 768 pixels.",
        )
        inference_steps = st.number_input(
            "Inference steps",
            min_value=1,
            max_value=10,
            value=DEFAULT_INFERENCE_STEPS,
            step=1,
            key="roughness_inference_steps",
        )
        ensemble_size = st.select_slider(
            "Ensemble size",
            options=[1, 3, 5],
            value=DEFAULT_ENSEMBLE_SIZE,
            key="roughness_ensemble_size",
            help="Three or more samples produce a predictive uncertainty map.",
        )
        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=2_147_483_647,
            value=2024,
            step=1,
            key="roughness_seed",
        )
        remove_background = st.checkbox(
            "Remove background before analysis",
            value=True,
            key="roughness_remove_background",
            help="Uses rembg with ISNet to isolate and tightly crop the original RGB object.",
        )
        crop_padding_ratio = st.slider(
            "Crop padding",
            min_value=0.0,
            max_value=0.5,
            value=DEFAULT_CROP_PADDING_RATIO,
            step=0.05,
            key="roughness_crop_padding_ratio",
        )
        contact_band_fraction = st.slider(
            "Central grasp-band fraction",
            min_value=0.2,
            max_value=1.0,
            value=DEFAULT_CONTACT_BAND_FRACTION,
            step=0.05,
            key="roughness_contact_band_fraction",
            help=(
                "Scores the central portion of the object's major axis. This removes caps "
                "and end faces, such as the top of a can, without assuming two tiny pads."
            ),
        )
        mask_erosion_ratio = st.slider(
            "Mask-edge erosion fraction",
            min_value=0.0,
            max_value=0.05,
            value=DEFAULT_MASK_EROSION_RATIO,
            step=0.005,
            key="roughness_mask_erosion_ratio",
            help="Removes segmentation halos before computing statistics.",
        )
        base_surface_sigma_ratio = st.slider(
            "Topography base-shape smoothing",
            min_value=0.01,
            max_value=0.12,
            value=DEFAULT_BASE_SURFACE_SIGMA_RATIO,
            step=0.01,
            key="topography_base_surface_sigma_ratio",
            disabled=TOPOGRAPHY not in analyses,
            help=(
                "Controls the scale treated as broad object shape. Local normal variation "
                "smaller than this scale remains in the bump residual."
            ),
        )
        st.caption(
            "The first use downloads the selected Marigold checkpoint(s) and the optional "
            "background-removal weights."
        )

    manual_bboxes: dict[str, tuple[int, int, int, int]] = {}
    if manual_region and selections:
        st.subheader("Manual analysis regions")
        st.caption(
            "Adjust the horizontal and vertical ranges; the red rectangle is the only "
            "region included in the reported statistics."
        )
        for selection in selections:
            with st.expander(selection.display_label, expanded=len(selections) == 1):
                manual_bboxes[selection.target_key] = _manual_region_controls(selection)

    run_requested = st.button(
        "Run Marigold",
        type="primary",
        width="stretch",
        disabled=not selections or not analyses,
        key="run_marigold",
    )
    if run_requested and selections and analyses:
        device = available_device()
        remover = (
            _background_remover(DEFAULT_BACKGROUND_MODEL)
            if remove_background and not manual_region
            else None
        )
        results_by_target: dict[str, dict[str, Any]] = {}
        errors: list[str] = []
        completed_results = 0
        requested_results = len(selections) * len(analyses)
        progress = st.progress(0.0, text="Preparing Marigold batch...")
        for selection in selections:
            analysis_image = selection.image
            manual_mask: Image.Image | None = None
            manual_rationale: str | None = None
            if manual_region:
                bbox = manual_bboxes[selection.target_key]
                analysis_image, manual_mask, padded_bbox = _prepare_manual_region(
                    selection.image,
                    bbox,
                    float(crop_padding_ratio),
                )
                manual_rationale = (
                    f"User-selected original-image ROI xyxy={bbox}; "
                    f"padded inference crop xyxy={padded_bbox}."
                )
            results: dict[str, dict[str, Any]] = {}
            if ROUGHNESS in analyses:
                progress.progress(
                    completed_results / requested_results,
                    text=f"Appearance roughness: {selection.display_label}",
                )
                try:
                    results[ROUGHNESS] = run_marigold(
                        _roughness_analyzer(device, int(processing_resolution)),
                        analysis_image,
                        selection.roughness_root,
                        background_remover=remover,
                        source_label=selection.source_label,
                        dataset_id=selection.dataset_id,
                        object_id=selection.object_id,
                        source_path=selection.source_path,
                        num_inference_steps=int(inference_steps),
                        ensemble_size=int(ensemble_size),
                        seed=int(seed),
                        crop_padding_ratio=float(crop_padding_ratio),
                        contact_band_fraction=float(contact_band_fraction),
                        mask_erosion_ratio=float(mask_erosion_ratio),
                        scoring_mask_source=manual_mask,
                        scoring_mask_rationale=manual_rationale,
                        run_key=STABLE_RUN_KEY,
                    )
                except Exception as error:  # optional model failures belong in the UI
                    errors.append(f"{selection.display_label} · Roughness: {error}")
                completed_results += 1
            if TOPOGRAPHY in analyses:
                progress.progress(
                    completed_results / requested_results,
                    text=f"Topography: {selection.display_label}",
                )
                try:
                    results[TOPOGRAPHY] = run_marigold_topography(
                        _topography_analyzer(device, int(processing_resolution)),
                        analysis_image,
                        selection.topography_root,
                        background_remover=remover,
                        source_label=selection.source_label,
                        dataset_id=selection.dataset_id,
                        object_id=selection.object_id,
                        source_path=selection.source_path,
                        num_inference_steps=int(inference_steps),
                        ensemble_size=int(ensemble_size),
                        seed=int(seed),
                        crop_padding_ratio=float(crop_padding_ratio),
                        contact_band_fraction=float(contact_band_fraction),
                        mask_erosion_ratio=float(mask_erosion_ratio),
                        base_surface_sigma_ratio=float(base_surface_sigma_ratio),
                        scoring_mask_source=manual_mask,
                        scoring_mask_rationale=manual_rationale,
                        run_key=STABLE_RUN_KEY,
                    )
                except Exception as error:  # optional model failures belong in the UI
                    errors.append(f"{selection.display_label} · Topography: {error}")
                completed_results += 1
            if results:
                results_by_target[selection.target_key] = {
                    "display_label": selection.display_label,
                    "results": results,
                }
        progress.progress(1.0, text="Marigold batch complete.")
        if results_by_target:
            st.session_state["marigold_last_results"] = {
                "results_by_target": results_by_target,
            }
            successful_results = sum(
                len(entry["results"]) for entry in results_by_target.values()
            )
            st.success(
                f"Updated {successful_results} Marigold result"
                f"{'s' if successful_results != 1 else ''} across "
                f"{len(results_by_target)} image"
                f"{'s' if len(results_by_target) != 1 else ''}."
            )
        for error in errors:
            st.error(error)

    payload = st.session_state.get("marigold_last_results")
    results_by_target = payload.get("results_by_target", {}) if isinstance(payload, dict) else {}
    for selection in selections:
        entry = results_by_target.get(selection.target_key, {})
        results = entry.get("results", {}) if isinstance(entry, dict) else {}
        if isinstance(results, dict) and results:
            if len(selections) > 1:
                st.divider()
                st.subheader(f"Results for {selection.display_label}")
            if isinstance(results.get(ROUGHNESS), dict):
                _render_roughness_run(results[ROUGHNESS])
            if isinstance(results.get(TOPOGRAPHY), dict):
                _render_topography_run(results[TOPOGRAPHY])


def render(context: AppContext) -> None:
    st.header("Marigold Roughness")
    st.write(
        "Run either or both Marigold analyses independently of the force pipeline. "
        "Roughness estimates material appearance; topography uses predicted surface normals "
        "to retain local bumps and grooves after subtracting broad object curvature. Results "
        "use an eroded central grasp band and update the same per-object files on rerun."
    )
    _run_workbench(context)
