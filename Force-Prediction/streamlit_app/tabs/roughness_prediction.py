"""Marigold roughness experimentation and saved-run browser."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image, UnidentifiedImageError

from modules.models.background_remover import (
    DEFAULT_BACKGROUND_MODEL,
    BackgroundRemover,
)
from modules.models.marigold import (
    MarigoldAnalyzer,
    available_device,
    list_saved_runs,
    run_marigold,
)
from streamlit_app.context import AppContext


@st.cache_resource(show_spinner=False)
def _analyzer(device: str, processing_resolution: int) -> MarigoldAnalyzer:
    return MarigoldAnalyzer(device=device, processing_resolution=processing_resolution)


@st.cache_resource(show_spinner=False)
def _background_remover(model_name: str) -> BackgroundRemover:
    return BackgroundRemover(model_name=model_name)


def _open_upload(uploaded: Any) -> Image.Image | None:
    if uploaded is None:
        return None
    try:
        return Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
    except (OSError, UnidentifiedImageError):
        return None


def _artifact_path(run: dict[str, Any], name: str) -> Path | None:
    artifact = run.get("artifacts", {}).get(name)
    if not artifact:
        return None
    path = Path(str(run["run_dir"])) / str(artifact)
    return path if path.is_file() else None


def _render_run(run: dict[str, Any]) -> None:
    source = run.get("source", {})
    model = run.get("model", {})
    st.subheader(source.get("label") or run.get("run_id", "Marigold run"))
    st.caption(
        f"{run.get('created_at', 'Unknown time')} · {model.get('id', 'Unknown model')} · "
        f"{model.get('device', 'unknown device')} · {run.get('duration_seconds', 0):g} s"
    )

    source_columns = st.columns(4, gap="medium")
    for column, artifact, caption in zip(
        source_columns,
        ("input", "foreground_preview", "cutout", "foreground_mask"),
        ("Input", "Foreground preview", "Transparent cutout", "Foreground mask"),
        strict=True,
    ):
        path = _artifact_path(run, artifact)
        if path:
            with column:
                st.image(str(path), caption=caption, width="stretch")

    image_columns = st.columns(3, gap="medium")
    for column, artifact, caption in zip(
        image_columns,
        ("albedo", "roughness", "metallicity"),
        ("Albedo", "Roughness (bright = rough)", "Metallicity (bright = metallic)"),
        strict=True,
    ):
        path = _artifact_path(run, artifact)
        with column:
            if path:
                st.image(str(path), caption=caption, width="stretch")
            else:
                st.warning(f"{caption} artifact is missing.")

    roughness = run.get("roughness", {})
    metallicity = run.get("metallicity", {})
    metrics = st.columns(4)
    metrics[0].metric("Mean roughness", f"{roughness.get('mean', float('nan')):.3f}")
    metrics[1].metric("Median roughness", f"{roughness.get('median', float('nan')):.3f}")
    metrics[2].metric("Roughness std.", f"{roughness.get('std', float('nan')):.3f}")
    metrics[3].metric("Mean metallicity", f"{metallicity.get('mean', float('nan')):.3f}")

    with st.expander("Run data and provenance"):
        display = {key: value for key, value in run.items() if key != "run_dir"}
        st.json(display, expanded=True)
        st.caption(f"Saved artifacts: {run['run_dir']}")
        metadata = json.dumps(display, indent=2).encode()
        st.download_button(
            "Download metadata JSON",
            data=metadata,
            file_name=f"{run.get('run_id', 'marigold_run')}_metadata.json",
            mime="application/json",
            key=f"roughness_download_{run.get('run_id', 'unknown')}",
        )


def _history(output_root: Path) -> None:
    runs = list_saved_runs(output_root)
    if not runs:
        st.info("No Marigold runs have been saved yet. Switch off history mode to create one.")
        return
    labels = {
        (
            f"{run.get('created_at', '')[:19]} | "
            f"{run.get('source', {}).get('label', run.get('run_id', 'run'))} | "
            f"{run.get('run_id', 'run')}"
        ): run
        for run in runs
    }
    selected = st.selectbox(
        "Previous Marigold run",
        list(labels),
        key="roughness_history_selector",
    )
    _render_run(labels[selected])


def _run_new(context: AppContext, output_root: Path) -> None:
    rows = sorted(context.rows, key=lambda row: (row.name.casefold(), row.object_id))
    labels = {
        f"{row.name} ({row.object_id})": row
        for row in rows
    }
    selected_row = None
    if labels:
        selected_label = st.selectbox(
            "Marigold dataset image",
            list(labels),
            key="roughness_dataset_image",
        )
        selected_row = labels[selected_label]
    else:
        st.info("This dataset contains no indexed images. Upload an override image below.")

    uploaded = st.file_uploader(
        "Override image",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
        key="roughness_override_image",
    )
    upload_image = _open_upload(uploaded)
    if uploaded is not None and upload_image is None:
        st.error("The uploaded file could not be decoded as an image.")

    dataset_path = None
    dataset_image = None
    if selected_row is not None:
        dataset_path = context.config.root / selected_row.image.path
        if dataset_path.is_file():
            try:
                dataset_image = Image.open(dataset_path).convert("RGB")
            except (OSError, UnidentifiedImageError):
                st.warning("The selected dataset image could not be decoded.")
        else:
            st.warning("The selected dataset image is not available locally.")

    query_image = upload_image or dataset_image
    if query_image is not None:
        caption = "Override image" if upload_image is not None else "Dataset image"
        st.image(query_image, caption=caption, width="stretch")

    with st.expander("Marigold settings"):
        processing_resolution = st.select_slider(
            "Processing resolution",
            options=[384, 512, 640, 768],
            value=640,
            key="roughness_processing_resolution",
            help="Lower resolutions use less memory. The reference script uses 640 pixels.",
        )
        inference_steps = st.number_input(
            "Inference steps",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
            key="roughness_inference_steps",
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
            "Remove background before computing roughness statistics",
            value=True,
            key="roughness_remove_background",
            help=(
                "Uses rembg with ISNet to create a foreground mask. Marigold still sees the "
                "original RGB image, while displayed maps and summary statistics are masked."
            ),
        )
        st.caption(
            "The first run downloads and loads the Marigold and background-removal weights."
        )

    run_requested = st.button(
        "Run Marigold",
        type="primary",
        width="stretch",
        disabled=query_image is None,
        key="run_marigold",
    )
    if run_requested and query_image is not None:
        try:
            device = available_device()
            source_label = (
                uploaded.name
                if uploaded is not None
                else selected_row.name if selected_row is not None else "image"
            )
            with st.spinner(f"Loading and running Marigold on {device}..."):
                result = run_marigold(
                    _analyzer(device, int(processing_resolution)),
                    query_image,
                    output_root,
                    background_remover=(
                        _background_remover(DEFAULT_BACKGROUND_MODEL)
                        if remove_background
                        else None
                    ),
                    source_label=source_label,
                    dataset_id=context.dataset.dataset_id if uploaded is None else None,
                    object_id=(
                        selected_row.object_id
                        if uploaded is None and selected_row is not None
                        else None
                    ),
                    source_path=(
                        str(selected_row.image.path)
                        if uploaded is None and selected_row is not None
                        else None
                    ),
                    num_inference_steps=int(inference_steps),
                    seed=int(seed),
                )
            st.session_state["roughness_last_run"] = result
            st.success(f"Saved Marigold run to {result['run_dir']}")
        except Exception as error:  # optional dependency/model failures belong in the UI
            st.error(str(error))

    result = st.session_state.get("roughness_last_run")
    if isinstance(result, dict):
        _render_run(result)


def render(context: AppContext) -> None:
    st.header("Marigold Roughness")
    st.write(
        "Test Marigold IID appearance decomposition independently of the force pipeline. "
        "Background removal isolates the object before foreground roughness statistics are "
        "computed. Each run stores the input, mask, cutout, albedo, roughness and metallicity "
        "maps, raw numeric arrays, summary statistics, and model provenance."
    )
    output_root = context.config.root / "test_data" / "marigold_tests"
    view_history = st.toggle(
        "View previous Marigold runs",
        value=False,
        key="roughness_view_history",
    )
    if view_history:
        _history(output_root)
    else:
        _run_new(context, output_root)
