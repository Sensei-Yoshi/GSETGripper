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
        "Run only the preparation stages needed for the active dataset. Missing prerequisites "
        "are filled automatically; downstream stages are never run unless selected."
    )
    st.info(
        "Descriptors and embeddings are checkpointed per object. Re-running a stage skips "
        "artifacts whose image, prompt, model, schema, and dimensions are still current."
    )

    stats = st.columns(4)
    stats[0].metric("Dataset", dataset.display_name)
    stats[1].metric("Objects", summary["objects"])
    stats[2].metric("Prepared descriptions", len(dataset.descriptions))
    stats[3].metric("Ready embeddings", sum(
        item.status == "ready" for item in dataset.embeddings.values()
    ))
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

    execution = st.segmented_control(
        "Preparation execution",
        ["Live Gemini", "Offline stubs"],
        default="Live Gemini",
        key="preparation_execution",
    )
    columns = st.columns(3)
    descriptions = columns[0].checkbox(
        "Gemini descriptions",
        value=True,
        key="prepare_descriptions",
        help="Indexes images first, generates or reuses structured descriptions, then stops.",
    )
    embeddings = columns[1].checkbox(
        "Text embeddings",
        value=False,
        key="prepare_embeddings",
        help="Generates missing descriptions first, then warms document embeddings.",
    )
    experiences = columns[2].checkbox(
        "Experience records",
        value=False,
        disabled=not dataset.capabilities.can_build_experiences,
        key="prepare_experiences",
        help=(
            "Requires mass, roughness, projected contact, and paired gripper labels."
        ),
    )
    if not dataset.capabilities.can_build_experiences:
        st.caption(
            "Experience records are unavailable because this dataset is image-only or lacks "
            "complete paired measurements and labels."
        )

    run = st.button(
        "Run selected preparation stages",
        type="primary",
        width="stretch",
        disabled=not any((descriptions, embeddings, experiences)),
        key="run_preparation_stages",
    )
    if run:
        selected: list[PreparationStage] = []
        if descriptions:
            selected.append(PreparationStage.DESCRIPTIONS)
        if embeddings:
            selected.append(PreparationStage.EMBEDDINGS)
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
                live=execution == "Live Gemini",
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
