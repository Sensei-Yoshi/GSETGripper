"""Dataset-scoped, stage-selectable preparation tab."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from modules.datasets import PreparationStage, prepare_dataset_stages
from streamlit_app.context import AppContext


def render(context: AppContext) -> None:
    dataset = context.dataset
    summary = context.summary
    st.subheader("Dataset preparation")
    st.write(
        "Build any combination of semantic, material, and geometry artifacts for the active "
        "dataset. Missing semantic prerequisites are filled automatically; unrelated stages "
        "are never run unless selected."
    )
    st.info(
        "Descriptors and embeddings are checkpointed per object. Re-running a stage skips "
        "artifacts whose image, prompt, model, schema, and dimensions are still current."
    )

    stats = st.columns(6)
    stats[0].metric("Dataset", dataset.display_name)
    stats[1].metric("Objects", summary["objects"])
    stats[2].metric("Second views", summary["second_images"])
    stats[3].metric("Descriptions", len(dataset.descriptions))
    stats[4].metric("Embeddings", sum(
        item.status == "ready" for item in dataset.embeddings.values()
    ))
    stats[5].metric(
        "Marigold results",
        sum(item.roughness is not None for item in dataset.objects.values()),
    )
    st.caption(f"Source fingerprint: `{summary['source_sha256']}`")

    distributions = [
        {"dimension": "Roughness", "category": str(category), "count": count}
        for category, count in summary["roughness_counts"].items()
    ] + [
        {"dimension": "Favored gripper", "category": category, "count": count}
        for category, count in summary["favored_counts"].items()
    ]
    if distributions:
        st.dataframe(pd.DataFrame(distributions), hide_index=True, width="stretch")

    st.caption(
        "Descriptions and embeddings use Gemini with content-hash caching. Marigold and "
        "the surface/contact estimator run locally."
    )
    st.markdown("#### Semantic preparation")
    descriptions = st.checkbox(
        "Gemini descriptions",
        value=True,
        key="prepare_descriptions",
        help="Indexes images first, generates or reuses structured descriptions, then stops.",
    )
    embeddings = st.checkbox(
        "Text embeddings",
        value=False,
        key="prepare_embeddings",
        help="Generates missing descriptions first, then warms document embeddings.",
    )
    st.markdown("#### Material and geometry preparation")
    roughness = st.checkbox(
        "Marigold roughness",
        value=False,
        disabled=not dataset.capabilities.has_images,
        key="prepare_roughness",
        help=(
            "Runs foreground-masked Marigold IID on each primary image and stores "
            "dataset-scoped maps, statistics, and provenance."
        ),
    )
    surface_area = st.checkbox(
        "Surface/contact fraction from image_2",
        value=False,
        disabled=not dataset.capabilities.can_estimate_surface_area,
        key="prepare_surface_area",
        help=(
            "Runs the calibrated projected two-pad contact estimator on image_2 for every "
            "object. This is a dimensionless contact proxy, not absolute area in mm²."
        ),
    )
    st.markdown("#### Training artifacts")
    experiences = st.checkbox(
        "Experience records",
        value=False,
        disabled=not dataset.capabilities.can_build_experiences,
        key="prepare_experiences",
        help=(
            "Builds one reference record per completed gripper outcome. Physical fields may "
            "remain blank for E3."
        ),
    )
    if not dataset.capabilities.can_build_experiences:
        st.caption(
            "Experience records become available after at least one gripper outcome is "
            "completed in Data Viewer."
        )
    if not dataset.capabilities.can_estimate_surface_area:
        st.caption(
            "Surface-area estimation is unavailable for this dataset: every object needs a "
            "second view. Use `objects/<object_id>/image_2.<ext>`, an `Image_2`/`image_2` CSV "
            "column, or paired flat files named `<object>` and `<object>_2`."
        )

    run = st.button(
        "Run selected preparation stages",
        type="primary",
        width="stretch",
        disabled=not any(
            (descriptions, embeddings, roughness, surface_area, experiences)
        ),
        key="run_preparation_stages",
    )
    if run:
        selected: list[PreparationStage] = []
        if descriptions:
            selected.append(PreparationStage.DESCRIPTIONS)
        if embeddings:
            selected.append(PreparationStage.EMBEDDINGS)
        if roughness:
            selected.append(PreparationStage.ROUGHNESS)
        if surface_area:
            selected.append(PreparationStage.SURFACE_AREA)
        if experiences:
            selected.append(PreparationStage.EXPERIENCES)
        progress_bar = st.progress(0.0)
        status = st.empty()

        def progress(done: int, total: int, name: str) -> None:
            progress_bar.progress(done / max(total, 1))
            status.caption(f"{done}/{total}: {name}")

        try:
            manifest = prepare_dataset_stages(
                context.config,
                dataset,
                selected,
                progress=progress,
            )
        except Exception as error:  # noqa: BLE001
            st.error(f"Preparation failed: {error}")
        else:
            st.session_state["preparation_manifest"] = manifest
            st.success(
                f"Finished selected stages for {dataset.display_name}. Cached artifacts will "
                "be reused by later preparation and pipeline runs."
            )
        progress_bar.empty()
        status.empty()

    if dataset.paths.preparation_manifest.exists():
        st.subheader("Preparation manifest")
        st.json(
            json.loads(dataset.paths.preparation_manifest.read_text()),
            expanded=False,
        )
