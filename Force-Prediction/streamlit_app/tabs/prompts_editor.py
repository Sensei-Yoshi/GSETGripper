"""Editable prompt and fixed gripper-embodiment context."""

from __future__ import annotations

import streamlit as st

from modules.config import (
    Config,
    EmbodimentContext,
    PromptBundle,
    Prompts,
    prompt_bundle_path,
    prompt_bundle_sha256,
    save_prompt_bundle,
)
from streamlit_app.context import AppContext

_LOADED_BUNDLE_SHA_KEY = "_prompt_editor_loaded_bundle_sha256"


def _sync_widgets_from_config(cfg: Config) -> str:
    """Refresh editor widgets when the persisted prompt bundle changes."""
    bundle_sha = prompt_bundle_sha256(cfg)
    if st.session_state.get(_LOADED_BUNDLE_SHA_KEY) == bundle_sha:
        return bundle_sha

    st.session_state.update(
        {
            "prompt_prediction_system": cfg.prompts.prediction_system,
            "prompt_descriptor_system": cfg.prompts.descriptor_system,
            "prompt_descriptor_instruction": cfg.prompts.descriptor,
            **{
                f"prompt_instruction_{experiment}": cfg.prompts.experiments[experiment]
                for experiment in ("e1", "e2", "e3", "e4")
            },
            **{
                f"embodiment_description_{name}": cfg.embodiments[name].description
                for name in ("gecko", "silicone")
            },
        }
    )
    st.session_state[_LOADED_BUNDLE_SHA_KEY] = bundle_sha
    return bundle_sha


def _bundle_from_widgets(context: AppContext) -> PromptBundle:
    experiments = {
        experiment: st.session_state[f"prompt_instruction_{experiment}"]
        for experiment in ("e1", "e2", "e3", "e4")
    }
    return PromptBundle(
        prompts=Prompts(
            descriptor_system=st.session_state["prompt_descriptor_system"],
            prediction_system=st.session_state["prompt_prediction_system"],
            descriptor=st.session_state["prompt_descriptor_instruction"],
            experiments=experiments,
        ),
        embodiments={
            name: EmbodimentContext(
                description=st.session_state[f"embodiment_description_{name}"],
            )
            for name in ("gecko", "silicone")
        },
    )


def render(context: AppContext) -> None:
    cfg = context.config
    bundle_sha = _sync_widgets_from_config(cfg)
    st.header("Prompts & Embodiments")
    st.caption(
        f"Saved in {cfg.prompts_file} | SHA-256 {bundle_sha[:16]}… | "
        "Save applies to Streamlit and CLI runs after validation."
    )

    st.subheader("Shared prompts")
    st.text_area(
        "Prediction system prompt",
        height=360,
        key="prompt_prediction_system",
    )
    descriptor_cols = st.columns(2)
    with descriptor_cols[0]:
        st.text_area(
            "Descriptor system prompt",
            height=260,
            key="prompt_descriptor_system",
        )
    with descriptor_cols[1]:
        st.text_area(
            "Descriptor instruction",
            height=260,
            key="prompt_descriptor_instruction",
        )

    st.subheader("Experiment instructions")
    for experiment in ("e1", "e2", "e3", "e4"):
        st.text_area(
            f"{experiment.upper()} instruction",
            height=170,
            key=f"prompt_instruction_{experiment}",
        )

    st.subheader("Fixed gripper context")
    columns = st.columns(2, gap="large")
    for column, name in zip(columns, ("gecko", "silicone"), strict=True):
        with column:
            st.markdown(f"**{name.title()} embodiment**")
            st.text_area(
                f"{name.title()} description",
                height=220,
                key=f"embodiment_description_{name}",
            )

    validate, save = st.columns(2)
    validate_clicked = validate.button("Validate prompt bundle", key="validate_prompt_bundle")
    save_clicked = save.button(
        "Save prompts & embodiments", type="primary", key="save_prompt_bundle"
    )
    if validate_clicked or save_clicked:
        try:
            bundle = _bundle_from_widgets(context)
        except ValueError as error:
            st.error(f"Validation failed: {error}")
            return
        if save_clicked:
            save_prompt_bundle(bundle, prompt_bundle_path(cfg))
            st.success("Saved prompt bundle. Reloading the application snapshot…")
            st.rerun()
        else:
            st.success("Prompt bundle is valid.")
