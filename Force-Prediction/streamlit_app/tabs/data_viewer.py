"""Dataset catalog plus the reusable inspector rendered by Runs Viewer."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from modules.artifacts import (
    load_experience_pool,
    load_saved_runs,
    pipeline_result_from_dict,
)
from modules.contracts import Gripper, group_by_object
from modules.datasets import (
    DatasetObject,
    DatasetObjectEdit,
    add_dataset_condition,
    delete_dataset_condition,
    update_dataset_object,
)
from streamlit_app.context import AppContext
from streamlit_app.prediction_ui import render_prediction

_SPLIT_ORDER = ("train", "test", "surface_validation")


def _split_label(split: str) -> str:
    return split.replace("_", " ").title()


def _surface_splits(conditions: list[DatasetObject]) -> tuple[str, ...]:
    present = {item.split for item in conditions}
    return tuple(split for split in _SPLIT_ORDER if split in present)


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
) -> None:
    prefix = f"data_viewer_{context.dataset.dataset_id}_{item.object_id}"

    def save_change() -> None:
        try:
            statuses: dict[str, bool | None] = {}
            forces: dict[str, float | None] = {}
            active = {gripper.value for gripper in context.config.prediction.active_grippers}
            for gripper in ("gecko", "silicone"):
                outcome = item.gripper_outcomes.get(Gripper(gripper))
                if gripper in active:
                    statuses[gripper] = {
                        "Unrecorded": None,
                        "Feasible": True,
                        "Infeasible": False,
                    }[st.session_state[f"{prefix}_{gripper}_status"]]
                    forces[gripper] = (
                        st.session_state[f"{prefix}_{gripper}_force_n"]
                        if statuses[gripper] is True
                        else None
                    )
                else:
                    statuses[gripper] = outcome.feasible if outcome else None
                    forces[gripper] = outcome.min_force_n if outcome else None
            for gripper, status in statuses.items():
                if status is not True:
                    st.session_state[f"{prefix}_{gripper}_force_n"] = None
            edit = DatasetObjectEdit(
                split=st.session_state[f"{prefix}_split"].lower().replace(" ", "_"),
                mass_g=st.session_state[f"{prefix}_mass_g"],
                roughness_index=(
                    st.session_state[f"{prefix}_roughness_index"]
                    if context.config.inputs.use_roughness
                    else item.roughness_index
                ),
                projected_contact_fraction=(
                    st.session_state[f"{prefix}_projected_contact_fraction"]
                    if context.config.inputs.use_projected_contact
                    else item.projected_contact_fraction
                ),
                gecko_feasible=statuses["gecko"],
                gecko_force_n=forces["gecko"],
                silicone_feasible=statuses["silicone"],
                silicone_force_n=forces["silicone"],
            )
            update_dataset_object(
                context.config,
                context.dataset,
                item.object_id,
                edit,
            )
        except (KeyError, OSError, ValueError) as error:
            st.session_state[f"{prefix}_save_status"] = f"error:{error}"
        else:
            st.session_state[f"{prefix}_save_status"] = "saved"

    with st.expander(f"Edit {item.condition_id.replace('_', ' ')}"):
        st.caption(
            "Each valid change saves atomically and refreshes completed experience records. "
            "The dataset split and measurements are editable; names, images, "
            "descriptors, and generated artifacts remain read-only."
        )
        split_labels = [_split_label(split) for split in _SPLIT_ORDER]
        st.selectbox(
            "Dataset split",
            split_labels,
            index=split_labels.index(_split_label(item.split)),
            key=f"{prefix}_split",
            on_change=save_change,
            help=(
                "Train rows may be used as E2-E5 references. Test rows are the only "
                "rows predicted and scored by a fixed-holdout benchmark. Surface "
                "validation rows are reserved for single-run surface checks and are "
                "excluded from benchmark train/test aggregates."
            ),
        )
        measurement_count = 1 + int(context.config.inputs.use_roughness) + int(
            context.config.inputs.use_projected_contact
        )
        measurement_cols = st.columns(measurement_count)
        measurement_cols[0].number_input(
            "Mass (g)",
            min_value=0.001,
            value=float(item.mass_g) if item.mass_g is not None else None,
            placeholder="Not recorded",
            key=f"{prefix}_mass_g",
            on_change=save_change,
        )
        measurement_index = 1
        if context.config.inputs.use_roughness:
            measurement_cols[measurement_index].number_input(
                "Roughness index",
                min_value=0.0,
                value=(
                    float(item.roughness_index)
                    if item.roughness_index is not None
                    else None
                ),
                step=0.01,
                format="%.2f",
                placeholder="Not recorded",
                key=f"{prefix}_roughness_index",
                on_change=save_change,
                help="Continuous LED sensor value; larger values indicate rougher surfaces.",
            )
            measurement_index += 1
        if context.config.inputs.use_projected_contact:
            measurement_cols[measurement_index].number_input(
                "Projected contact fraction",
                min_value=0.0,
                max_value=1.0,
                value=(
                    float(item.projected_contact_fraction)
                    if item.projected_contact_fraction is not None
                    else None
                ),
                step=0.001,
                format="%.3f",
                placeholder="Not recorded",
                key=f"{prefix}_projected_contact_fraction",
                on_change=save_change,
            )

        active_grippers = tuple(
            gripper.value for gripper in context.config.prediction.active_grippers
        )
        outcome_cols = st.columns(len(active_grippers))
        for column, gripper in zip(outcome_cols, active_grippers, strict=True):
            outcome = item.gripper_outcomes.get(Gripper(gripper))
            status = (
                "Unrecorded"
                if outcome is None or outcome.feasible is None
                else "Feasible"
                if outcome.feasible
                else "Infeasible"
            )
            with column:
                st.markdown(f"**{gripper.title()} outcome**")
                selected_status = st.selectbox(
                    "Status",
                    ["Unrecorded", "Feasible", "Infeasible"],
                    index=["Unrecorded", "Feasible", "Infeasible"].index(status),
                    key=f"{prefix}_{gripper}_status",
                    on_change=save_change,
                )
                st.number_input(
                    "Minimum force (N)",
                    min_value=0.001,
                    value=(
                        float(outcome.min_force_n)
                        if outcome is not None and outcome.min_force_n is not None
                        else None
                    ),
                    step=0.05,
                    placeholder="Enter force to complete a feasible outcome",
                    disabled=selected_status != "Feasible",
                    key=f"{prefix}_{gripper}_force_n",
                    on_change=save_change,
                )

        saved = st.session_state.get(f"{prefix}_save_status")
        if saved == "saved":
            st.success("Saved automatically.")
        elif isinstance(saved, str) and saved.startswith("error:"):
            st.error("Could not save: " + saved.removeprefix("error:"))

        if item.condition_id != "baseline":
            st.divider()
            confirmed = st.checkbox(
                "I understand this permanently removes this condition row.",
                key=f"{prefix}_confirm_delete",
            )
            if st.button(
                "Delete condition",
                disabled=not confirmed,
                key=f"{prefix}_delete",
            ):
                try:
                    delete_dataset_condition(
                        context.config, context.dataset, item.object_id
                    )
                except (KeyError, OSError, ValueError) as error:
                    st.error(str(error))
                else:
                    st.rerun()


@st.dialog("Add measurement condition")
def _add_condition_dialog(context: AppContext, baseline: DatasetObject) -> None:
    descriptor = (
        baseline.description.value.retrieval_description
        if baseline.description
        else "Not prepared"
    )
    embedding = baseline.embedding.status if baseline.embedding else "pending"
    st.info(
        f"Inherited read-only artifacts: {Path(baseline.image.path).name}; "
        f"descriptor: {descriptor}; embedding: {embedding}."
    )
    with st.form(f"add_condition_{context.dataset.dataset_id}_{baseline.surface_id}"):
        mass = st.number_input(
            "Mass (g) *",
            min_value=0.001,
            value=float(baseline.mass_g or 0.001),
        )
        roughness = None
        if context.config.inputs.use_roughness:
            roughness = st.number_input(
                "Recorded LED roughness index *",
                min_value=0.0,
                value=float(baseline.roughness_index or 0.0),
            )
        contact = None
        if context.config.inputs.use_projected_contact:
            contact = st.number_input(
                "Projected contact fraction (0–1) *",
                min_value=0.0,
                max_value=1.0,
                value=float(baseline.projected_contact_fraction or 0.0),
                step=0.001,
                format="%.3f",
            )
        outcome_values: dict[str, tuple[bool | None, float | None]] = {}
        columns = st.columns(len(context.config.prediction.active_grippers))
        for column, gripper in zip(
            columns, context.config.prediction.active_grippers, strict=True
        ):
            with column:
                status_label = st.selectbox(
                    f"{gripper.value.title()} outcome",
                    ["Unrecorded", "Feasible", "Infeasible"],
                )
                status = {
                    "Unrecorded": None,
                    "Feasible": True,
                    "Infeasible": False,
                }[status_label]
                force = st.number_input(
                    f"{gripper.value.title()} minimum force (N)",
                    min_value=0.001,
                    value=None,
                    disabled=status is not True,
                )
                outcome_values[gripper.value] = (
                    status,
                    force if status is True else None,
                )
        submitted = st.form_submit_button("Add condition", type="primary")
    if not submitted:
        return
    gecko = outcome_values.get("gecko", (None, None))
    silicone = outcome_values.get("silicone", (None, None))
    try:
        _, object_id = add_dataset_condition(
            context.config,
            context.dataset,
            baseline.surface_id or baseline.object_id,
            DatasetObjectEdit(
                mass_g=mass,
                roughness_index=roughness,
                projected_contact_fraction=contact,
                gecko_feasible=gecko[0],
                gecko_force_n=gecko[1],
                silicone_feasible=silicone[0],
                silicone_force_n=silicone[1],
            ),
        )
    except (KeyError, OSError, ValueError) as error:
        st.error(str(error))
    else:
        st.session_state["data_viewer_flash"] = f"Added {object_id}."
        st.rerun()


def _description_catalog(context: AppContext) -> None:
    base_cfg = context.config
    rows = context.rows
    grouped: dict[str, list[DatasetObject]] = {}
    for row in rows:
        grouped.setdefault(row.surface_id or row.object_id, []).append(row)
    for conditions in grouped.values():
        conditions.sort(
            key=lambda item: (item.condition_id != "baseline", item.condition_id)
        )
    surfaces = [
        next(
            (item for item in conditions if item.condition_id == "baseline"),
            conditions[0],
        )
        for conditions in grouped.values()
    ]
    selected_split_labels = st.multiselect(
        "Filter by dataset split",
        [_split_label(split) for split in _SPLIT_ORDER],
        default=[_split_label(split) for split in _SPLIT_ORDER],
        key=f"catalog_split_filter_{context.dataset.dataset_id}",
        help=(
            "A physical object is included when any of its measurement conditions has "
            "a selected split. Its card still shows every condition."
        ),
    )
    selected_splits = {
        label.lower().replace(" ", "_") for label in selected_split_labels
    }
    search = st.text_input("Search objects", placeholder="Material, object, condition...")
    page_size = st.segmented_control(
        "Objects per page", [8, 12, 24], default=12, key="catalog_page_size"
    )
    needle = search.strip().lower()
    filtered = []
    for row in surfaces:
        conditions = grouped[row.surface_id or row.object_id]
        if not selected_splits.intersection(_surface_splits(conditions)):
            continue
        description = row.description.value if row.description else None
        searchable = " ".join(
            (
                row.name,
                description.retrieval_description if description else "",
                description.contact_material if description else "",
                description.visible_surface_condition if description else "",
                *(
                    f"{condition.object_id} {condition.condition_id} {condition.split}"
                    for condition in conditions
                ),
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
        f"of {len(filtered)} physical surfaces. "
        f"{len(rows)} measurement conditions are available."
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
                        image_2 = _thumbnail(str(image_2_path), image_2_path.stat().st_mtime_ns)
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
                surface_id = row.surface_id or row.object_id
                conditions = grouped[surface_id]
                split_summary = ", ".join(
                    _split_label(split) for split in _surface_splits(conditions)
                )
                st.caption(
                    f"Surface ID: {surface_id} · Dataset splits: {split_summary} · "
                    f"{len(conditions)} measurement condition(s)"
                )

                st.markdown("**Measurement conditions**")
                condition_rows = []
                for condition in conditions:
                    condition_row = {
                        "Condition": condition.condition_id,
                        "Data-point ID": condition.object_id,
                        "Split": _split_label(condition.split),
                        "Mass (g)": condition.mass_g,
                    }
                    if base_cfg.inputs.use_roughness:
                        condition_row["LED roughness"] = condition.roughness_index
                    if base_cfg.inputs.use_projected_contact:
                        condition_row["Contact fraction"] = (
                            condition.projected_contact_fraction
                        )
                    for gripper in base_cfg.prediction.active_grippers:
                        outcome = condition.gripper_outcomes.get(gripper)
                        condition_row[f"{gripper.value.title()} outcome"] = (
                            f"{outcome.min_force_n:g} N"
                            if outcome and outcome.min_force_n is not None
                            else "Infeasible"
                            if outcome and outcome.feasible is False
                            else "Unrecorded"
                        )
                    condition_rows.append(condition_row)
                st.dataframe(condition_rows, hide_index=True, width="stretch")
                if st.button(
                    "+ Add condition",
                    key=f"add_condition_{context.dataset.dataset_id}_{surface_id}",
                    disabled=context.dataset.adapter != "paired_csv",
                ):
                    _add_condition_dialog(context, row)

                if row.roughness is not None:
                    st.caption(
                        f"Marigold roughness: mean {row.roughness.mean:.3f}, "
                        f"median {row.roughness.median:.3f}, std. {row.roughness.std:.3f}"
                    )
                    if row.roughness.quality_status == "warning":
                        st.warning("Marigold quality: " + ", ".join(row.roughness.quality_warnings))

                for condition in conditions:
                    _card_editor(context, condition)

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
                st.write(descriptor.retrieval_description)
                st.markdown(
                    f"**Contact material:** {descriptor.contact_material}  \n"
                    f"**Surface condition:** "
                    f"{descriptor.visible_surface_condition}  \n"
                    f"**Material visibility:** {descriptor.contact_patch_visibility}  \n"
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
        st.metric(
            "Mass",
            f"{query['mass_g']:.1f} g" if query.get("mass_g") is not None else "Not recorded",
        )
        sensor_cols = st.columns(2)
        roughness_display = (
            f"{query['roughness_index']:g}"
            if query.get("roughness_index") is not None
            else "Not recorded"
        )
        sensor_cols[0].metric("Roughness index", roughness_display)
        sensor_cols[1].metric(
            "Contact",
            f"{query['projected_contact_fraction']:.3f}"
            if query.get("projected_contact_fraction") is not None
            else "Not recorded",
        )
        st.subheader("Run configuration")
        st.write(f"**Experiment:** {run['experiment_display_name']}")
        st.write(
            f"**Method/version:** {run['experiment_method']} / "
            f"{run['experiment_definition_version']}"
        )
        st.write(f"**Backend:** {run['backend_label']}")
        st.write(f"**VLM:** {run['models']['vlm']}")
        st.write(f"**Text embedding:** {run['models']['embedding']}")
        st.write(f"**Protocol:** {run['evaluation_protocol'].replace('-', ' ')}")
        active_grippers = tuple(run.get("active_grippers", ("gecko", "silicone")))
        st.write(f"**Active grippers:** {', '.join(active_grippers)}")
        st.write(f"**Description:** {query.get('semantic_description', '')}")
        with st.expander("Retrieval parameters"):
            retrieval = run["retrieval_config"]
            saved_inputs = run.get("inputs", {})
            contact_enabled = saved_inputs.get(
                "use_projected_contact",
                retrieval.get("use_projected_contact", True),
            )
            st.write(f"Roughness enabled: {saved_inputs.get('use_roughness', True)}")
            st.write("VLM roughness representation: continuous index")
            st.write(f"Projected contact enabled: {contact_enabled}")
            st.write(f"Saved run top k: {retrieval['k']}")
            if retrieval["k"] != base_cfg.retrieval.k:
                st.info(
                    "This historical run used "
                    f"k={retrieval['k']}; the current config.yaml uses "
                    f"k={base_cfg.retrieval.k}. New runs use the current value."
                )
            st.write(f"Mass sigma: {retrieval['sigma_mass']}")
            st.json(retrieval["weights"], expanded=True)
        with st.expander("Experiment definition"):
            st.json(
                run.get("experiment_definition", run.get("experiment_toggles", {})),
                expanded=True,
            )
        truth = run.get("truth")
        if truth:
            st.subheader("Saved synthetic truth")
            truth_parts = [
                f"{name.title()}: {truth.get(f'true_{name}_force_n')} N"
                for name in active_grippers
            ]
            if truth.get("true_selection"):
                truth_parts.append(f"Winner: {truth['true_selection'].title()}")
            st.write(" | ".join(truth_parts))
            if run.get("counterfactual"):
                st.caption("Context only: counterfactual runs are not scored against this truth.")

    with output_col:
        st.subheader("Pipeline output")
        cfg = base_cfg.model_copy(deep=True)
        cfg.retrieval = type(cfg.retrieval).model_validate(run["retrieval_config"])
        cfg.inputs = type(cfg.inputs).model_validate(run.get("inputs", {}))
        if run.get("roughness_measurement"):
            cfg.roughness = type(cfg.roughness).model_validate(
                run["roughness_measurement"]
            )
        cfg.prediction.active_grippers = tuple(
            Gripper(name) for name in run.get("active_grippers", ("gecko", "silicone"))
        )
        detailed = pipeline_result_from_dict(run["result"])
        baseline = pipeline_result_from_dict(run["baseline"]) if run.get("baseline") else None
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
    splits_by_surface: dict[str, set[str]] = {}
    for item in context.rows:
        splits_by_surface.setdefault(item.surface_id or item.object_id, set()).add(item.split)
    split_counts = Counter(
        split for surface_splits in splits_by_surface.values() for split in surface_splits
    )
    st.caption(
        f"Dataset split membership · Train: {split_counts['train']} physical surfaces · "
        f"Test: {split_counts['test']} physical surfaces · Surface validation: "
        f"{split_counts['surface_validation']} physical surfaces"
    )
    if message := st.session_state.pop("data_viewer_flash", None):
        st.success(message)
    _description_catalog(context)
