"""Application shell for the Streamlit force-prediction research lab."""

from __future__ import annotations

import streamlit as st

from modules.config import load_config
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

    def clear_dataset_state() -> None:
        for key in (
            "single_result",
            "benchmark_result",
            "contact_last",
            "roughness_last_run",
            "marigold_last_results",
            "preparation_manifest",
            "last_experiment",
        ):
            st.session_state.pop(key, None)

    choices = [dataset.dataset_id for dataset in catalog]
    selector_col, roughness_col, contact_col = st.columns([0.5, 0.25, 0.25])
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
    with roughness_col:
        use_roughness = st.checkbox(
            "Consider roughness class",
            value=base_cfg.inputs.use_roughness,
            key="consider_roughness",
            on_change=clear_dataset_state,
            help="Controls E2 inputs and the E4 hybrid retrieval term.",
        )
    with contact_col:
        use_projected_contact = st.checkbox(
            "Consider projected contact fraction",
            value=base_cfg.inputs.use_projected_contact,
            key="consider_projected_contact",
            on_change=clear_dataset_state,
            help="Controls E2 inputs and the E4 hybrid retrieval term.",
        )
    base_cfg.inputs.use_roughness = use_roughness
    base_cfg.inputs.use_projected_contact = use_projected_contact
    context = load_context(selected, base_config=base_cfg, catalog=catalog)
    capability = context.dataset.capabilities

    st.caption(
        f"{len(context.dataset.objects)} objects · {context.dataset.adapter.replace('_', ' ')} · "
        f"descriptions {'ready' if capability.has_descriptions else 'not prepared'} · "
        f"embeddings {'ready' if capability.has_embeddings else 'not prepared'}"
    )
    tab_containers = st.tabs([spec.label for spec in TAB_SPECS])
    for container, spec in zip(tab_containers, TAB_SPECS, strict=True):
        with container:
            spec.render(context)
