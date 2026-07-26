"""Editable prompt and fixed gripper-embodiment context."""

from __future__ import annotations

import streamlit as st

from modules.config import (
    EmbodimentContext,
    PromptBundle,
    Prompts,
    prompt_bundle_path,
    prompt_bundle_sha256,
    save_prompt_bundle,
)
from streamlit_app.context import AppContext


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
    st.header("Prompts & Embodiments")
    st.caption(
        f"Saved in {cfg.prompts_file} | SHA-256 {prompt_bundle_sha256(cfg)[:16]}… | "
        "Save applies to Streamlit and CLI runs after validation."
    )

    st.subheader("Shared prompts")
    st.text_area(
        "Prediction system prompt",
        value=cfg.prompts.prediction_system,
        height=360,
        key="prompt_prediction_system",
    )
    descriptor_cols = st.columns(2)
    with descriptor_cols[0]:
        st.text_area(
            "Descriptor system prompt",
            value=cfg.prompts.descriptor_system,
            height=260,
            key="prompt_descriptor_system",
        )
    with descriptor_cols[1]:
        st.text_area(
            "Descriptor instruction",
            value=cfg.prompts.descriptor,
            height=260,
            key="prompt_descriptor_instruction",
        )

    st.subheader("Experiment instructions")
    for experiment in ("e1", "e2", "e3", "e4"):
        st.text_area(
            f"{experiment.upper()} instruction",
            value=cfg.prompts.experiments[experiment],
            height=170,
            key=f"prompt_instruction_{experiment}",
        )

    st.subheader("Fixed gripper context")
    columns = st.columns(2, gap="large")
    for column, name in zip(columns, ("gecko", "silicone"), strict=True):
        with column:
            embodiment = cfg.embodiments[name]
            st.markdown(f"**{name.title()} embodiment**")
            st.text_area(
                f"{name.title()} description",
                value=embodiment.description,
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
