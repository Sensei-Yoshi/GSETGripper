"""Single-query pipeline tab."""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

from modules.config import EXPERIMENT_IDS, Config
from modules.contracts import group_by_object
from modules.experiments import EXPERIMENT_CATALOG
from modules.expforce import (
    load_experience_pool,
    load_image,
    save_pipeline_run,
)
from modules.pipeline import Pipeline, QueryInput
from modules.retrieval import normalized_weights
from streamlit_app.context import AppContext
from streamlit_app.prediction_ui import (
    format_experiment,
    render_formula,
    render_prediction,
    render_semantic_formula,
    truth_payload,
)


def _run_config(
    base: Config,
    *,
    live: bool,
    use_projected_contact: bool,
    semantic: float,
    mass: float,
    roughness: float,
    contact: float,
    sigma_mass: float,
    sigma_contact: float,
    validate_hybrid_weights: bool,
) -> Config:
    cfg = base.model_copy(deep=True)
    cfg.models.dry_run = not live
    cfg.inputs.use_projected_contact = use_projected_contact
    cfg.retrieval.weights.semantic = semantic
    cfg.retrieval.weights.mass = mass
    cfg.retrieval.weights.roughness = roughness
    cfg.retrieval.weights.contact = contact
    cfg.retrieval.sigma_mass = sigma_mass
    cfg.retrieval.sigma_contact = sigma_contact
    if validate_hybrid_weights:
        normalized_weights(cfg)
    return cfg


def _decode_upload(uploaded) -> np.ndarray | None:  # noqa: ANN001
    if uploaded is None:
        return None
    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def render(context: AppContext) -> None:
    base_cfg = context.config
    rows = context.rows
    if not context.dataset.capabilities.can_run_pipeline:
        st.warning(
            f"{context.dataset.display_name} is available for viewing and semantic preparation, "
            "but Single Run requires mass, roughness, projected contact, and paired gripper labels."
        )
        return
    records = load_experience_pool(base_cfg)
    objects = group_by_object(records)
    dataset_objects = {row.object_id: row for row in rows}
    names = {row.name: row.object_id for row in rows}

    controls, output = st.columns([0.38, 0.62], gap="large")
    with controls:
        st.subheader("Query")
        selected_name = st.selectbox("Dataset object", sorted(names))
        object_id = names[selected_name]
        dataset_object = dataset_objects[object_id]
        truth = objects[object_id]
        sample = truth.gecko or truth.silicone
        assert sample is not None
        source_image = load_image(base_cfg, sample)
        uploaded = st.file_uploader("Override image", type=["png", "jpg", "jpeg"])
        upload_image = _decode_upload(uploaded)
        query_image = upload_image if upload_image is not None else source_image
        if query_image is not None:
            st.image(_rgb(query_image), width="stretch")
        else:
            st.warning("Image is not available locally. Prepare images before a live run.")

        experiment = st.selectbox(
            "Experiment profile",
            list(EXPERIMENT_IDS),
            index=3,
            format_func=format_experiment,
        )
        uses_measurements = EXPERIMENT_CATALOG[experiment].uses_measurements
        uses_hybrid_retrieval = experiment == "e4"
        mass = st.number_input(
            "Mass (g)",
            min_value=0.1,
            value=float(sample.mass_g),
            step=1.0,
            disabled=not uses_measurements,
            help="This authoritative value is hidden from E1 and E3.",
        )
        roughness = st.select_slider(
            "Roughness class",
            options=[1, 2, 3, 4, 5],
            value=int(sample.roughness_class),
            disabled=not uses_measurements,
        )
        contact = st.slider(
            "Projected contact fraction",
            min_value=0.0,
            max_value=1.0,
            value=float(sample.projected_contact_fraction),
            step=0.001,
            disabled=not uses_measurements,
        )
        if not uses_measurements:
            st.caption(
                f"{experiment.upper()} does not expose mass, roughness, or projected "
                "contact to the estimator."
            )
        mode = st.segmented_control("Execution", ["Offline", "Live Gemini"], default="Offline")
        live_execution = mode == "Live Gemini"

        with st.expander("Input and retrieval tuning", expanded=True):
            use_projected_contact = st.checkbox(
                "Use projected contact fraction",
                value=base_cfg.inputs.use_projected_contact,
                disabled=not uses_measurements,
                help=(
                    "When disabled, contact is omitted from VLM inputs and its retrieval "
                    "weight is set to zero; the remaining weights are renormalized. "
                    "Physics-based E5 and E6 still require the measured contact fraction."
                ),
            )
            semantic_w = st.slider(
                "Semantic weight", 0.0, 1.0,
                float(base_cfg.retrieval.weights.semantic), 0.05,
                disabled=not uses_hybrid_retrieval,
            )
            mass_w = st.slider(
                "Mass weight", 0.0, 1.0,
                float(base_cfg.retrieval.weights.mass), 0.05,
                disabled=not uses_hybrid_retrieval,
            )
            roughness_w = st.slider(
                "Roughness weight", 0.0, 1.0,
                float(base_cfg.retrieval.weights.roughness), 0.05,
                disabled=not uses_hybrid_retrieval,
            )
            configured_contact_w = st.slider(
                "Contact weight", 0.0, 1.0,
                float(base_cfg.retrieval.weights.contact), 0.05,
                disabled=not use_projected_contact or not uses_hybrid_retrieval,
            )
            contact_w = configured_contact_w if use_projected_contact else 0.0
            sigma_mass = st.slider(
                "Mass sigma", 0.1, 3.0,
                float(base_cfg.retrieval.sigma_mass), 0.1,
                disabled=not uses_hybrid_retrieval,
            )
            sigma_contact = st.slider(
                "Contact sigma", 0.05, 1.0,
                float(base_cfg.retrieval.sigma_contact), 0.05,
                disabled=not use_projected_contact or not uses_hybrid_retrieval,
            )
            st.caption(f"Neighbor count comes from config.yaml: k = {base_cfg.retrieval.k}.")

        run = st.button("Run pipeline", type="primary", width="stretch")

    try:
        cfg = _run_config(
            base_cfg,
            live=live_execution,
            use_projected_contact=use_projected_contact,
            semantic=semantic_w,
            mass=mass_w,
            roughness=roughness_w,
            contact=contact_w,
            sigma_mass=sigma_mass,
            sigma_contact=sigma_contact,
            validate_hybrid_weights=uses_hybrid_retrieval,
        )
    except ValueError as error:
        with output:
            st.error(str(error))
        return

    if run:
        counterfactual = bool(
            uploaded is not None
            or (
                uses_measurements
                and (
                    not np.isclose(mass, sample.mass_g)
                    or roughness != sample.roughness_class
                    or not np.isclose(contact, sample.projected_contact_fraction)
                )
            )
        )
        training = (
            records
            if counterfactual
            else [record for record in records if record.object_id != object_id]
        )
        prepared_text = (
            dataset_object.description.value.description
            if dataset_object.description is not None
            else None
        )
        semantic_description = (
            None if uploaded is not None else prepared_text or sample.semantic_description
        )
        with output, st.spinner("Running the shared pipeline..."):
            pipe = Pipeline(cfg, experiment).fit(training)
            detailed = pipe.predict_detailed(
                QueryInput(
                    object_id=f"custom_{object_id}" if counterfactual else object_id,
                    mass_g=mass,
                    roughness_class=roughness,
                    projected_contact_fraction=contact,
                    image_bgr=query_image,
                    image_path=sample.image_path if uploaded is None else "",
                    semantic_description=semantic_description,
                )
            )
            baseline = None
            if counterfactual:
                baseline_pipe = Pipeline(cfg, experiment).fit(
                    [record for record in records if record.object_id != object_id]
                )
                baseline = baseline_pipe.predict_detailed(
                    QueryInput(
                        object_id=object_id,
                        mass_g=sample.mass_g,
                        roughness_class=sample.roughness_class,
                        projected_contact_fraction=sample.projected_contact_fraction,
                        image_bgr=source_image,
                        image_path=sample.image_path,
                        semantic_description=prepared_text or sample.semantic_description,
                    )
                )
            run_path = save_pipeline_run(
                cfg,
                detailed=detailed,
                experiment=experiment,
                execution_mode=mode or "Offline",
                query={
                    "object_id": f"custom_{object_id}" if counterfactual else object_id,
                    "source_object_id": object_id,
                    "object_name": selected_name,
                    "mass_g": mass,
                    "roughness_class": roughness,
                    "projected_contact_fraction": contact,
                    "semantic_description": detailed.semantic_description,
                    "original_image_path": sample.image_path if uploaded is None else None,
                },
                truth=truth_payload(truth),
                counterfactual=counterfactual,
                image_bgr=query_image,
                baseline=baseline,
            )
            st.session_state["single_result"] = (
                detailed,
                truth,
                counterfactual,
                baseline,
                cfg,
                experiment,
                run_path,
            )
            st.session_state["last_experiment"] = experiment

    with output:
        st.subheader("Pipeline output")
        if "single_result" not in st.session_state:
            st.info("Select a dataset object or upload an image, then run the pipeline.")
            if experiment == "e3":
                render_semantic_formula()
            elif experiment == "e4":
                render_formula(cfg)
            else:
                st.caption(f"{experiment.upper()} does not use experiential retrieval.")
        else:
            (
                detailed,
                stored_truth,
                counterfactual,
                baseline,
                stored_cfg,
                stored_experiment,
                run_path,
            ) = st.session_state["single_result"]
            render_prediction(
                detailed,
                stored_truth,
                counterfactual=counterfactual,
                baseline=baseline,
                cfg=stored_cfg,
                experiment=stored_experiment,
            )
            st.caption(f"Saved run: {run_path.name}")
