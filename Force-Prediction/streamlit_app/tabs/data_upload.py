"""Upload new physical objects and baseline measurements to the active dataset."""

from __future__ import annotations

import streamlit as st

from modules.datasets import DatasetObjectEdit, add_dataset_object
from streamlit_app.context import AppContext


def _outcome(prefix: str, label: str) -> tuple[bool | None, float | None]:
    status = st.selectbox(
        f"{label} status",
        ["Unrecorded", "Feasible", "Infeasible"],
        key=f"{prefix}_{label.lower()}_status",
    )
    force = st.number_input(
        f"{label} minimum force (N)",
        min_value=0.001,
        value=None,
        placeholder="Required when feasible",
        key=f"{prefix}_{label.lower()}_force",
    )
    feasible = {"Unrecorded": None, "Feasible": True, "Infeasible": False}[status]
    return feasible, force if feasible is True else None


def render(context: AppContext) -> None:
    st.header("Upload Data")
    st.write(
        "Add a new physical object to the selected dataset. The images, baseline "
        "measurements, split, and gripper outcomes are saved together."
    )
    st.caption(
        "Uploaded images are stored under the dataset's canonical `objects/` folder. "
        "You can leave measurements or outcomes blank and complete them later in Data Viewer."
    )

    flash = st.session_state.pop("data_upload_flash", None)
    if flash:
        st.success(flash)

    generation = int(st.session_state.get("data_upload_generation", 0))
    prefix = f"data_upload_{context.dataset.dataset_id}_{generation}"
    with st.form(f"{prefix}_form", clear_on_submit=False):
        name_col, split_col = st.columns([0.7, 0.3])
        with name_col:
            object_name = st.text_input(
                "Object name",
                placeholder="e.g. Ceramic coffee mug",
                key=f"{prefix}_name",
            )
        with split_col:
            split = st.selectbox(
                "Upload split",
                ["Train", "Test"],
                key=f"{prefix}_split",
                help="All later conditions for this physical surface inherit this split.",
            )

        image_col, image_2_col = st.columns(2)
        with image_col:
            primary = st.file_uploader(
                "Primary dataset image",
                type=["png", "jpg", "jpeg", "webp"],
                key=f"{prefix}_primary",
                help="Used for Gemini descriptions, embeddings, and Marigold.",
            )
        with image_2_col:
            secondary = st.file_uploader(
                "Geometry image (optional)",
                type=["png", "jpg", "jpeg", "webp"],
                key=f"{prefix}_secondary",
                help="A calibrated second view used for projected-contact estimation.",
            )

        st.markdown("**Baseline measurements**")
        measurement_cols = st.columns(3)
        mass = measurement_cols[0].number_input(
            "Upload mass (g)",
            min_value=0.001,
            value=None,
            placeholder="Not recorded",
            key=f"{prefix}_mass",
        )
        roughness = measurement_cols[1].number_input(
            "Upload roughness index",
            min_value=0.0,
            value=None,
            placeholder="Not recorded",
            key=f"{prefix}_roughness",
        )
        contact = measurement_cols[2].number_input(
            "Upload projected contact fraction",
            min_value=0.0,
            max_value=1.0,
            value=None,
            placeholder="Not recorded",
            key=f"{prefix}_contact",
        )

        st.markdown("**Gripper outcomes**")
        gecko_col, silicone_col = st.columns(2)
        with gecko_col:
            gecko_feasible, gecko_force = _outcome(prefix, "Gecko")
        with silicone_col:
            silicone_feasible, silicone_force = _outcome(prefix, "Silicone")

        submitted = st.form_submit_button(
            "Add object to dataset",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return
    if primary is None:
        st.error("Choose a primary image before adding the object.")
        return
    try:
        _, object_id = add_dataset_object(
            context.config,
            context.dataset,
            object_name,
            primary.getvalue(),
            DatasetObjectEdit(
                split=split.lower(),
                mass_g=mass,
                roughness_index=roughness,
                projected_contact_fraction=contact,
                gecko_feasible=gecko_feasible,
                gecko_force_n=gecko_force,
                silicone_feasible=silicone_feasible,
                silicone_force_n=silicone_force,
            ),
            primary_filename=primary.name,
            secondary_image=secondary.getvalue() if secondary is not None else None,
            secondary_filename=secondary.name if secondary is not None else None,
        )
    except (KeyError, OSError, ValueError) as error:
        st.error(str(error))
        return

    st.session_state["data_upload_flash"] = (
        f"Added {object_name.strip()} as `{object_id}` to {context.dataset.display_name}."
    )
    st.session_state["data_upload_generation"] = generation + 1
    st.rerun()
