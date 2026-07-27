"""Dataset catalog plus the reusable inspector rendered by Runs Viewer."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from modules.contracts import Gripper, group_by_object
from modules.datasets import DatasetObject, DatasetObjectEdit, update_csv_dataset_object
from modules.expforce import (
    ExpForceRow,
    load_experience_pool,
    load_rows,
    load_saved_runs,
    pipeline_result_from_dict,
)
from streamlit_app.context import AppContext
from streamlit_app.prediction_ui import render_prediction


def _rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


@st.cache_data(show_spinner=False)
def _thumbnail(path: str, modified_ns: int, max_width: int = 420) -> np.ndarray | None:
    del modified_ns
    image = cv2.imread(path)
    if image is None:
        return None
    if image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        image = cv2.resize(
            image,
            (max_width, max(1, int(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return _rgb(image)


def _card_editor(
    context: AppContext,
    item: DatasetObject,
    source: ExpForceRow,
) -> None:
    prefix = f"data_viewer_{context.dataset.dataset_id}_{item.object_id}"
    with st.expander("Edit measurements and outcomes"):
        st.caption(
            "Saves atomically to dataset.csv and refreshes paired experience records. "
            "Names, images, descriptors, and generated artifacts remain read-only."
        )
        with st.form(f"{prefix}_form"):
            measurement_cols = st.columns(3)
            mass_g = measurement_cols[0].number_input(
                "Mass (g)",
                min_value=0.001,
                value=float(source.mass_g),
                key=f"{prefix}_mass_g",
            )
            roughness_class = measurement_cols[1].number_input(
                "Roughness class",
                min_value=1,
                max_value=5,
                value=int(source.roughness_class),
                step=1,
                key=f"{prefix}_roughness_class",
            )
            projected_contact_fraction = measurement_cols[2].number_input(
                "Projected contact fraction",
                min_value=0.0,
                max_value=1.0,
                value=float(item.projected_contact_fraction or 0.0),
                step=0.001,
                format="%.3f",
                key=f"{prefix}_projected_contact_fraction",
            )

            outcome_cols = st.columns(2)
            outcome_values: dict[str, tuple[bool, float | None]] = {}
            for column, gripper in zip(
                outcome_cols, ("gecko", "silicone"), strict=True
            ):
                feasible = getattr(source, f"{gripper}_feasible")
                force = getattr(source, f"{gripper}_force_n")
                with column:
                    st.markdown(f"**{gripper.title()} outcome**")
                    edited_feasible = st.checkbox(
                        "Feasible",
                        value=feasible,
                        key=f"{prefix}_{gripper}_feasible",
                    )
                    edited_force = st.number_input(
                        "Minimum force (N)",
                        min_value=0.001,
                        max_value=float(context.config.force.limit_n),
                        value=float(force) if force is not None else None,
                        step=0.05,
                        placeholder="No force for an infeasible outcome",
                        key=f"{prefix}_{gripper}_force_n",
                    )
                    outcome_values[gripper] = (
                        edited_feasible,
                        float(edited_force)
                        if edited_feasible and edited_force is not None
                        else None,
                    )
            candidate_payload = {
                "silicone_force_n": outcome_values["silicone"][1],
                "silicone_feasible": outcome_values["silicone"][0],
                "gecko_force_n": outcome_values["gecko"][1],
                "gecko_feasible": outcome_values["gecko"][0],
            }
            st.caption(
                "Favored gripper is recalculated on save. "
                f"Current value: {source.favored_gripper.title()}"
            )

            submitted = st.form_submit_button(
                "Save object changes",
                type="primary",
                key=f"{prefix}_save",
            )
        if submitted:
            try:
                edit = DatasetObjectEdit(
                    mass_g=mass_g,
                    roughness_class=roughness_class,
                    projected_contact_fraction=projected_contact_fraction,
                    **candidate_payload,
                )
                update_csv_dataset_object(
                    context.config,
                    context.dataset,
                    item.object_id,
                    edit,
                )
            except (KeyError, OSError, ValueError) as error:
                st.error(f"Could not save object: {error}")
            else:
                st.success("Saved CSV measurements and experience records. Reloading…")
                st.rerun()


def _description_catalog(context: AppContext) -> None:
    base_cfg = context.config
    rows = context.rows
    source_rows = (
        {row.object_id: row for row in load_rows(context.config)}
        if context.dataset.adapter == "expforce_paired_csv"
        else {}
    )
    search = st.text_input("Search objects", placeholder="Material, object, condition...")
    page_size = st.segmented_control(
        "Objects per page", [8, 12, 24], default=12, key="catalog_page_size"
    )
    needle = search.strip().lower()
    filtered = []
    for row in rows:
        description = row.description.value if row.description else None
        searchable = " ".join(
            (
                row.name,
                description.description if description else "",
                description.contact_material if description else "",
                description.visible_surface_condition if description else "",
            )
        ).lower()
        if not needle or needle in searchable:
            filtered.append(row)

    size = int(page_size or 12)
    page_count = max(1, (len(filtered) + size - 1) // size)
    page = st.number_input(
        "Page", min_value=1, max_value=page_count, value=1, step=1, key="catalog_page"
    )
    start = (int(page) - 1) * size
    st.caption(
        f"Showing {start + 1 if filtered else 0}-{min(start + size, len(filtered))} "
        f"of {len(filtered)} objects. All {len(rows)} are available through search and paging."
    )

    for row in filtered[start : start + size]:
        image_path = base_cfg.root / row.image.path
        with st.container(border=True):
            image_col, detail_col = st.columns([0.28, 0.72], gap="medium")
            with image_col:
                st.markdown("**Image 1 — Perception view**")
                st.caption("Gemini, embeddings, and Marigold")
                if image_path.exists():
                    image = _thumbnail(str(image_path), image_path.stat().st_mtime_ns)
                    if image is not None:
                        st.image(image, width="stretch")
                    else:
                        st.warning("Primary image could not be decoded.")
                else:
                    st.warning("Primary image is unavailable.")
                st.caption(f"`{Path(row.image.path).name}`")

                st.markdown("**Image 2 — Geometry view**")
                st.caption("Surface/contact fraction")
                if row.image_2 is not None:
                    image_2_path = base_cfg.root / row.image_2.path
                    if image_2_path.exists():
                        image_2 = _thumbnail(
                            str(image_2_path), image_2_path.stat().st_mtime_ns
                        )
                        if image_2 is not None:
                            st.image(image_2, width="stretch")
                        else:
                            st.warning("Second image could not be decoded.")
                    else:
                        st.warning("Second image is unavailable.")
                    st.caption(f"`{Path(row.image_2.path).name}` (`image_2`)")
                else:
                    st.info("No `image_2` indexed.")
                    st.caption("Required for surface/contact estimation")

            with detail_col:
                st.subheader(row.name)
                st.caption(f"Object ID: {row.object_id}")

                sensor_cols = st.columns(3)
                sensor_cols[0].metric(
                    "Mass", f"{row.mass_g:g} g" if row.mass_g is not None else "Not available"
                )
                sensor_cols[1].metric(
                    "Roughness class",
                    row.roughness_class if row.roughness_class is not None else "Not available",
                )
                sensor_cols[2].metric(
                    "Projected contact fraction",
                    (
                        f"{row.projected_contact_fraction:.3f}"
                        if row.projected_contact_fraction is not None
                        else "Not available"
                    ),
                )
                if row.roughness is not None:
                    st.caption(
                        f"Marigold roughness: mean {row.roughness.mean:.3f}, "
                        f"median {row.roughness.median:.3f}, std. {row.roughness.std:.3f}"
                    )

                if row.gripper_outcomes:
                    force_cols = st.columns(2)
                    for column, gripper in zip(
                        force_cols, ("gecko", "silicone"), strict=True
                    ):
                        outcome = row.gripper_outcomes.get(Gripper(gripper))
                        column.metric(
                            f"{gripper.title()} force",
                            (
                                f"{outcome.min_force_n:.2f} N"
                                if outcome and outcome.min_force_n is not None
                                else "Infeasible" if outcome else "Not available"
                            ),
                        )
                        if outcome:
                            column.caption(f"Feasible: {'Yes' if outcome.feasible else 'No'}")

                source = source_rows.get(row.object_id)
                if source is not None:
                    _card_editor(context, row, source)

                if row.description is None:
                    st.warning("No descriptor checkpoint. Run live Data Preparation.")
                    continue
                descriptor = row.description.value
                embedding = row.embedding
                source_label = row.description.source.replace("_", " ").title()
                st.caption(
                    f"{source_label} | "
                    f"Embedding {embedding.status if embedding else 'pending'} | "
                    f"{embedding.model if embedding and embedding.model else 'not generated'}"
                )
                st.write(descriptor.description)
                st.markdown(
                    f"**Contact region:** {descriptor.contact_region}  \n"
                    f"**Contact material:** {descriptor.contact_material}  \n"
                    f"**Surface condition:** "
                    f"{descriptor.visible_surface_condition}  \n"
                    f"**Local geometry:** {descriptor.local_geometry}  \n"
                    f"**Uncertainty:** {descriptor.uncertainty}"
                )


def pipeline_run_inspector(context: AppContext) -> None:
    base_cfg = context.config
    runs = load_saved_runs(base_cfg)
    if not runs:
        st.info("No saved single runs yet. Run an object in Single Run first.")
        return

    labels = {
        (
            f"{run['created_at'][:19]} | {run['query'].get('object_name', run['query']['object_id'])} "
            f"| {run['experiment_display_name']} | {run['backend_label']}"
        ): run
        for run in runs
    }
    selected = st.selectbox("Saved run", list(labels), key="saved_run_selector")
    run = labels[selected]
    query = run["query"]
    current_hash = context.dataset.source_fingerprint
    if run.get("source_sha256") != current_hash:
        st.warning(
            "This run was produced from a different dataset version. Its saved truth is shown "
            "for provenance, but it is not rescored against the current data."
        )

    query_col, output_col = st.columns([0.34, 0.66], gap="large")
    with query_col:
        st.subheader("Exact query")
        image_rel = query.get("image_artifact_path") or query.get("original_image_path")
        image_path = base_cfg.root / image_rel if image_rel else None
        if image_path and image_path.exists():
            st.image(str(image_path), width="stretch")
        else:
            st.warning("Saved query image is unavailable.")
        st.caption(f"Image SHA-256: {query.get('image_sha256') or 'not recorded'}")
        st.metric("Mass", f"{query['mass_g']:.1f} g")
        sensor_cols = st.columns(2)
        sensor_cols[0].metric("Roughness", query["roughness_class"])
        sensor_cols[1].metric("Contact", f"{query['projected_contact_fraction']:.3f}")
        st.subheader("Run configuration")
        st.write(f"**Experiment:** {run['experiment_display_name']}")
        st.write(
            f"**Method/version:** {run.get('experiment_method', 'legacy')} / "
            f"{run.get('experiment_definition_version', 'legacy')}"
        )
        st.write(f"**Backend:** {run['backend_label']}")
        st.write(f"**VLM:** {run['models']['vlm']}")
        st.write(f"**Text embedding:** {run['models']['embedding']}")
        st.write(f"**Protocol:** {run['evaluation_protocol'].replace('-', ' ')}")
        st.write(f"**Description:** {query.get('semantic_description', '')}")
        with st.expander("Retrieval parameters"):
            retrieval = run["retrieval_config"]
            saved_inputs = run.get("inputs", {})
            contact_enabled = saved_inputs.get(
                "use_projected_contact",
                retrieval.get("use_projected_contact", True),
            )
            st.write(f"Projected contact enabled: {contact_enabled}")
            st.write(f"Saved run top k: {retrieval['k']}")
            if retrieval["k"] != base_cfg.retrieval.k:
                st.info(
                    "This historical run used "
                    f"k={retrieval['k']}; the current config.yaml uses "
                    f"k={base_cfg.retrieval.k}. New runs use the current value."
                )
            st.write(f"Mass sigma: {retrieval['sigma_mass']}")
            st.write(f"Contact sigma: {retrieval['sigma_contact']}")
            st.json(retrieval["weights"], expanded=True)
        with st.expander("Experiment definition"):
            st.json(
                run.get("experiment_definition", run.get("experiment_toggles", {})),
                expanded=True,
            )
        truth = run.get("truth")
        if truth:
            st.subheader("Saved synthetic truth")
            st.write(
                f"Gecko: {truth.get('true_gecko_force_n')} N | "
                f"Silicone: {truth.get('true_silicone_force_n')} N | "
                f"Winner: {truth.get('true_selection', 'unknown').title()}"
            )
            if run.get("counterfactual"):
                st.caption("Context only: counterfactual runs are not scored against this truth.")

    with output_col:
        st.subheader("Pipeline output")
        cfg = base_cfg.model_copy(deep=True)
        cfg.retrieval = type(cfg.retrieval).model_validate(run["retrieval_config"])
        cfg.inputs.use_projected_contact = run.get("inputs", {}).get(
            "use_projected_contact",
            run["retrieval_config"].get("use_projected_contact", True),
        )
        detailed = pipeline_result_from_dict(run["result"])
        baseline = (
            pipeline_result_from_dict(run["baseline"]) if run.get("baseline") else None
        )
        records = load_experience_pool(base_cfg)
        objects = group_by_object(records)
        source_id = query.get("source_object_id")
        truth_obj = objects.get(source_id)
        score_as_current = (
            truth_obj is not None
            and run.get("source_sha256") == current_hash
            and not run.get("counterfactual", False)
        )
        render_prediction(
            detailed,
            truth_obj,
            counterfactual=not score_as_current,
            baseline=baseline,
            cfg=cfg,
            experiment=run["experiment_display_name"],
        )


def render(context: AppContext) -> None:
    st.header("Data Viewer")
    _description_catalog(context)
