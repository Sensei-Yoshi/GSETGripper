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
            "preparation_manifest",
            "last_experiment",
        ):
            st.session_state.pop(key, None)

    choices = [dataset.dataset_id for dataset in catalog]
    selected = st.selectbox(
        "Dataset",
        choices,
        key="active_dataset_id",
        on_change=clear_dataset_state,
        format_func=lambda value: next(
            dataset.display_name for dataset in catalog if dataset.dataset_id == value
        ),
    )
    context = load_context(selected, base_config=base_cfg, catalog=catalog)
    capability = context.dataset.capabilities
    if capability.has_paired_labels:
        st.markdown(
            '<div class="synthetic-note"><b>Synthetic pipeline validation.</b> '
            "These labels test software behavior and model integration, not physical gripper performance.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "This is an image-only dataset. Data Viewer, Data Preparation, Contact Fraction, "
            "Marigold Roughness, and dataset-scoped cache inspection remain available."
        )
    st.caption(
        f"{len(context.dataset.objects)} objects · {context.dataset.adapter.replace('_', ' ')} · "
        f"descriptions {'ready' if capability.has_descriptions else 'not prepared'} · "
        f"embeddings {'ready' if capability.has_embeddings else 'not prepared'}"
    )
    tab_containers = st.tabs([spec.label for spec in TAB_SPECS])
    for container, spec in zip(tab_containers, TAB_SPECS, strict=True):
        with container:
            spec.render(context)
