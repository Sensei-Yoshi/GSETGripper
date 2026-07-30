"""Single-query pipeline tab."""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

from modules.config import EXPERIMENT_IDS, Config
from modules.contracts import ExperienceRecord, group_by_object
from modules.datasets import DatasetObject
from modules.experiments import EXPERIMENT_CATALOG, experiment_eligibility
from modules.expforce import (
    load_experience_pool,
    save_pipeline_run,
)
from modules.pipeline import Pipeline, QueryInput
from modules.retrieval import RankingFeature, normalized_weights
from modules.roughness_representation import binary_roughness_category
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
    semantic: float,
    mass: float,
    roughness: float,
    contact: float,
    sigma_mass: float,
    sigma_contact: float,
    validate_hybrid_weights: bool,
    ranking_features: tuple[RankingFeature, ...] | None,
) -> Config:
    cfg = base.model_copy(deep=True)
    cfg.retrieval.weights.semantic = semantic
    cfg.retrieval.weights.mass = mass
    cfg.retrieval.weights.roughness = roughness
    cfg.retrieval.weights.contact = contact
    cfg.retrieval.sigma_mass = sigma_mass
    cfg.retrieval.sigma_contact = sigma_contact
    if validate_hybrid_weights:
        normalized_weights(cfg, ranking_features)
    return cfg


def _decode_upload(uploaded) -> np.ndarray | None:  # noqa: ANN001
    if uploaded is None:
        return None
    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _object_label(row: DatasetObject) -> str:
    """Selector label that disambiguates sibling conditions of one surface.

    Objects sharing a physical surface share a display name (e.g. both hand
    sanitizer conditions), so key the selector on the condition and flag
    surface_validation objects, which are single-run only and never in the
    benchmark train/test aggregates.
    """
    label = row.name
    if row.condition_id and row.condition_id != "baseline":
        label = f"{label} — {row.condition_id}"
    if row.split == "surface_validation":
        label = f"{label} [surface validation]"
    return label


def _single_run_training_records(
    records: list[ExperienceRecord],
    *,
    experiment: str,
    query_object_id: str,
    query_surface_id: str | None,
    reference_ids: tuple[str, ...],
) -> list[ExperienceRecord]:
    """Build the fold-local pool without exposing held-out or sibling truth."""
    if EXPERIMENT_CATALOG[experiment].retrieval_mode is None:
        return [record for record in records if record.object_id != query_object_id]

    allowed = set(reference_ids)
    return [
        record
        for record in records
        if record.object_id in allowed and record.surface_id != query_surface_id
    ]


def render(context: AppContext) -> None:
    base_cfg = context.config
    rows = context.rows
    if not rows:
        st.warning(f"{context.dataset.display_name} has no indexed objects.")
        return
    records = load_experience_pool(base_cfg)
    objects = group_by_object(records)
    dataset_objects = {row.object_id: row for row in rows}
    names = {_object_label(row): row.object_id for row in rows}

    controls, output = st.columns([0.38, 0.62], gap="large")
    with controls:
        st.subheader("Query")
        selected_name = st.selectbox("Dataset object", sorted(names))
        object_id = names[selected_name]
        dataset_object = dataset_objects[object_id]
        source_path = base_cfg.root / dataset_object.image.path
        source_image = cv2.imread(str(source_path)) if source_path.is_file() else None
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
            index=len(EXPERIMENT_IDS) - 1,
            format_func=format_experiment,
        )
        experiment_spec = EXPERIMENT_CATALOG[experiment]
        uses_measurements = experiment_spec.uses_measurements
        uses_roughness = experiment_spec.uses_roughness
        uses_projected_contact = experiment_spec.uses_projected_contact
        uses_hybrid_retrieval = (
            experiment_spec.retrieval_mode is not None and experiment != "e3"
        )
        contact_ranks_surfaces = "contact" in experiment_spec.ranking_features
        eligibility = experiment_eligibility(context.dataset, base_cfg, experiment)
        query_reasons = eligibility.query_reasons(object_id)
        if query_reasons:
            st.warning("Cannot run this object: " + "; ".join(query_reasons) + ".")
        mass = st.number_input(
            "Mass (g)",
            min_value=0.1,
            value=(
                float(dataset_object.mass_g)
                if dataset_object.mass_g is not None
                else None
            ),
            step=1.0,
            disabled=not uses_measurements,
            help="This authoritative value is hidden from E1 and E3.",
        )
        roughness = st.number_input(
            "Roughness index",
            min_value=0.0,
            value=(
                float(dataset_object.roughness_index)
                if dataset_object.roughness_index is not None
                else None
            ),
            step=0.01,
            format="%.2f",
            placeholder="Not recorded",
            disabled=not uses_roughness,
            help="Continuous sensor value; larger values indicate rougher surfaces.",
        )
        if (
            uses_roughness
            and experiment_spec.scoped_config(base_cfg).inputs.roughness_representation
            == "binary"
            and roughness is not None
        ):
            st.caption(
                "Gemini receives only: **"
                + binary_roughness_category(
                    float(roughness),
                    base_cfg.roughness.binary_threshold,
                )
                + "**. The numerical index remains internal to retrieval."
            )
        contact = st.number_input(
            "Projected contact fraction",
            min_value=0.0,
            max_value=1.0,
            value=(
                float(dataset_object.projected_contact_fraction)
                if dataset_object.projected_contact_fraction is not None
                else None
            ),
            step=0.001,
            disabled=not uses_projected_contact,
        )
        if not uses_measurements:
            st.caption(
                f"{experiment.upper()} does not expose mass, roughness, or projected "
                "contact to the estimator."
            )
        elif experiment in {"e4", "e5", "e6"}:
            enabled_measurements = ["mass"]
            if uses_roughness:
                enabled_measurements.append("continuous roughness")
            if uses_projected_contact:
                enabled_measurements.append("projected contact")
            st.caption(
                f"{experiment.upper()} is a fixed ablation: "
                + ", ".join(enabled_measurements)
                + "."
            )
        with st.expander("Input and retrieval tuning", expanded=True):
            st.caption(
                "E4-E6 have fixed ranking subsets; only enabled weights affect "
                "cross-object surface ranking."
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
                disabled=not uses_hybrid_retrieval or not uses_roughness,
            )
            configured_contact_w = st.slider(
                "Contact weight", 0.0, 1.0,
                float(base_cfg.retrieval.weights.contact), 0.05,
                disabled=(
                    not uses_hybrid_retrieval or not contact_ranks_surfaces
                ),
            )
            contact_w = (
                configured_contact_w if contact_ranks_surfaces else 0.0
            )
            sigma_mass = st.slider(
                "Mass sigma", 0.1, 3.0,
                float(base_cfg.retrieval.sigma_mass), 0.1,
                disabled=not uses_hybrid_retrieval,
            )
            sigma_contact = st.slider(
                "Contact sigma", 0.05, 1.0,
                float(base_cfg.retrieval.sigma_contact), 0.05,
                disabled=(
                    not uses_hybrid_retrieval or not contact_ranks_surfaces
                ),
            )
            st.caption(f"Neighbor count comes from config.yaml: k = {base_cfg.retrieval.k}.")

        run = st.button(
            "Run pipeline",
            type="primary",
            width="stretch",
            disabled=bool(query_reasons) or query_image is None,
        )

    try:
        cfg = _run_config(
            base_cfg,
            semantic=semantic_w,
            mass=mass_w,
            roughness=(roughness_w if uses_roughness else 0.0),
            contact=contact_w,
            sigma_mass=sigma_mass,
            sigma_contact=sigma_contact,
            validate_hybrid_weights=uses_hybrid_retrieval,
            ranking_features=experiment_spec.ranking_features or None,
        )
    except ValueError as error:
        with output:
            st.error(str(error))
        return

    if run:
        truth = objects.get(object_id) if object_id in eligibility.benchmark_ids else None
        counterfactual = bool(
            uploaded is not None
            or (
                uses_measurements
                and (
                    mass != dataset_object.mass_g
                    or (
                        uses_roughness
                        and roughness != dataset_object.roughness_index
                    )
                    or (
                        uses_projected_contact
                        and contact != dataset_object.projected_contact_fraction
                    )
                )
            )
        )
        training = _single_run_training_records(
            records,
            experiment=experiment,
            query_object_id=object_id,
            query_surface_id=dataset_object.surface_id,
            reference_ids=eligibility.reference_ids,
        )
        prepared_text = (
            dataset_object.description.value.description
            if dataset_object.description is not None
            else None
        )
        semantic_description = (
            None if uploaded is not None else prepared_text
        )
        with output, st.spinner("Running the shared pipeline..."):
            pipe = Pipeline(cfg, experiment).fit(training)
            detailed = pipe.predict_detailed(
                QueryInput(
                    object_id=f"custom_{object_id}" if counterfactual else object_id,
                    surface_id=dataset_object.surface_id,
                    condition_id=dataset_object.condition_id,
                    mass_g=float(mass) if mass is not None else None,
                    roughness_index=(float(roughness) if roughness is not None else None),
                    projected_contact_fraction=(
                        float(contact) if contact is not None else None
                    ),
                    image_bgr=query_image,
                    image_path=dataset_object.image.path if uploaded is None else "",
                    semantic_description=semantic_description,
                )
            )
            baseline = None
            if counterfactual and object_id in eligibility.query_ids:
                baseline_pipe = Pipeline(cfg, experiment).fit(training)
                baseline = baseline_pipe.predict_detailed(
                    QueryInput(
                        object_id=object_id,
                        surface_id=dataset_object.surface_id,
                        condition_id=dataset_object.condition_id,
                        mass_g=dataset_object.mass_g,
                        roughness_index=dataset_object.roughness_index,
                        projected_contact_fraction=dataset_object.projected_contact_fraction,
                        image_bgr=source_image,
                        image_path=dataset_object.image.path,
                        semantic_description=prepared_text,
                    )
                )
            run_path = save_pipeline_run(
                cfg,
                detailed=detailed,
                experiment=experiment,
                query={
                    "object_id": f"custom_{object_id}" if counterfactual else object_id,
                    "source_object_id": object_id,
                    "surface_id": dataset_object.surface_id,
                    "condition_id": dataset_object.condition_id,
                    "object_name": selected_name,
                    "mass_g": mass,
                    "roughness_index": roughness,
                    "projected_contact_fraction": contact,
                    "semantic_description": detailed.semantic_description,
                    "original_image_path": (
                        dataset_object.image.path if uploaded is None else None
                    ),
                },
                truth=(
                    truth_payload(truth, detailed.active_grippers)
                    if truth is not None
                    else None
                ),
                counterfactual=counterfactual or truth is None,
                image_bgr=query_image,
                baseline=baseline,
            )
            st.session_state["single_result"] = (
                detailed,
                truth,
                counterfactual or truth is None,
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
            elif uses_hybrid_retrieval:
                render_formula(
                    experiment_spec.scoped_config(cfg),
                    experiment_spec.ranking_features or None,
                )
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
