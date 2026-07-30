"""Application shell for the Streamlit force-prediction research lab."""

from __future__ import annotations

import streamlit as st

from modules.config import load_config
from modules.contracts import Gripper
from modules.datasets import discover_datasets
from streamlit_app.context import load_context
from streamlit_app.style import apply_style


def main() -> None:
    st.set_page_config(page_title="Force Pipeline Lab", page_icon="FP", layout="wide")
    apply_style()

    # Import after page configuration so tab modules cannot precede set_page_config
    # with cache-decorator registration or any future Streamlit calls.
    from streamlit_app.tabs.registry import TAB_SPECS

    st.title("Force Pipeline Lab")
    # Streamlit reruns within one Python process, while prompts.yaml may also be
    # edited outside the in-app editor. Clear the process-local loader cache so
    # every rerun uses the prompt bundle currently persisted on disk.
    load_config.cache_clear()
    base_cfg = load_config().model_copy(deep=True)
    catalog = discover_datasets(base_cfg)
    if not catalog:
        st.error(f"No dataset folders were found under `{base_cfg.root / 'data'}`.")
        return

    def clear_run_state() -> None:
        for key in (
            "single_result",
            "benchmark_result",
            "benchmark_prediction_result",
            "benchmark_evaluation_result",
            "contact_last",
            "roughness_last_run",
            "marigold_last_results",
            "preparation_manifest",
            "last_experiment",
        ):
            st.session_state.pop(key, None)

    choices = [dataset.dataset_id for dataset in catalog]
    datasets_by_id = {dataset.dataset_id: dataset for dataset in catalog}

    def clear_dataset_state() -> None:
        clear_run_state()
        dataset = datasets_by_id[st.session_state["active_dataset_id"]]
        defaults = set(dataset.default_active_grippers())
        st.session_state["predict_gecko_force"] = Gripper.GECKO in defaults
        st.session_state["predict_silicone_force"] = Gripper.SILICONE in defaults

    (
        selector_col,
        roughness_col,
        roughness_mode_col,
        contact_col,
        gecko_col,
        silicone_col,
    ) = st.columns(
        [0.25, 0.16, 0.19, 0.20, 0.10, 0.10]
    )
    with selector_col:
        selected = st.selectbox(
            "Dataset",
            choices,
            key="active_dataset_id",
            on_change=clear_dataset_state,
            format_func=lambda value: next(
                dataset.display_name for dataset in catalog if dataset.dataset_id == value
            ),
        )
    selected_dataset = datasets_by_id[selected]
    if "predict_gecko_force" not in st.session_state:
        defaults = set(selected_dataset.default_active_grippers())
        st.session_state["predict_gecko_force"] = Gripper.GECKO in defaults
        st.session_state["predict_silicone_force"] = Gripper.SILICONE in defaults
    with roughness_col:
        st.caption("Sensor subsets are fixed by experiment (E4-E6).")
        use_roughness = True
    with roughness_mode_col:
        roughness_representation = st.selectbox(
            "Roughness sent to VLM",
            ["continuous", "binary"],
            index=(
                1
                if base_cfg.inputs.roughness_representation == "binary"
                else 0
            ),
            format_func=lambda value: (
                "Smooth/Rough (test)" if value == "binary" else "Continuous index"
            ),
            key="roughness_representation",
            on_change=clear_run_state,
            help=(
                "This switch applies to E2. E5 and E6 always receive the continuous "
                "roughness index so the ablation remains fixed."
            ),
        )
    with contact_col:
        st.caption("Projected contact is used only by E2 and E6.")
        use_projected_contact = True
    selectable = set(selected_dataset.selectable_grippers())
    if Gripper.GECKO not in selectable:
        st.session_state["predict_gecko_force"] = False
    if Gripper.SILICONE not in selectable:
        st.session_state["predict_silicone_force"] = False
    gecko_selected = bool(st.session_state["predict_gecko_force"])
    silicone_selected = bool(st.session_state["predict_silicone_force"])
    with gecko_col:
        predict_gecko = st.checkbox(
            "Predict Gecko force",
            key="predict_gecko_force",
            on_change=clear_run_state,
            disabled=(
                Gripper.GECKO not in selectable
                or (gecko_selected and not silicone_selected)
            ),
            help="Use Gecko as an active force-prediction candidate.",
        )
    with silicone_col:
        predict_silicone = st.checkbox(
            "Predict silicone force",
            key="predict_silicone_force",
            on_change=clear_run_state,
            disabled=(
                Gripper.SILICONE not in selectable
                or (silicone_selected and not gecko_selected)
            ),
            help="Use silicone as an active force-prediction candidate.",
        )
    base_cfg.inputs.use_roughness = use_roughness
    base_cfg.inputs.roughness_representation = roughness_representation
    base_cfg.inputs.use_projected_contact = use_projected_contact
    base_cfg.prediction.active_grippers = tuple(
        gripper
        for gripper, enabled in (
            (Gripper.GECKO, predict_gecko),
            (Gripper.SILICONE, predict_silicone),
        )
        if enabled
    )
    context = load_context(selected, base_config=base_cfg, catalog=catalog)
    capability = context.dataset.capabilities

    st.caption(
        f"{context.summary['physical_surfaces']} physical surfaces · "
        f"{context.summary['measurement_conditions']} measurement conditions · "
        f"{context.summary['unique_photos']} unique photos · "
        f"{context.dataset.adapter.replace('_', ' ')} · "
        f"descriptions {'ready' if capability.has_descriptions else 'not prepared'} · "
        f"embeddings {'ready' if capability.has_embeddings else 'not prepared'}"
        f" · active grippers {', '.join(g.value for g in context.config.prediction.active_grippers)}"
    )
    if context.summary["missing_photos"]:
        st.warning(
            f"{len(context.summary['missing_photos'])} indexed photo(s) are unavailable."
        )
    tab_containers = st.tabs([spec.label for spec in TAB_SPECS])
    for container, spec in zip(tab_containers, TAB_SPECS, strict=True):
        with container:
            spec.render(context)
