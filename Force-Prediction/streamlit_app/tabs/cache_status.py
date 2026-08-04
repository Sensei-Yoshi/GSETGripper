"""API cache and run-artifact status tab."""

from __future__ import annotations

import streamlit as st

from modules.artifacts import load_saved_runs
from streamlit_app.context import AppContext


def render(context: AppContext) -> None:
    base_cfg = context.config
    cache_dir = base_cfg.path("cache")
    embedding_files = list((cache_dir / "embeddings").glob("*.json"))
    generation_files = list((cache_dir / "generation").glob("*.json"))
    result_dir = context.dataset.paths.results
    results = list(result_dir.glob("*")) if result_dir.exists() else []
    runs = load_saved_runs(base_cfg)
    stats = st.columns(4)
    stats[0].metric("Cached embeddings", len(embedding_files))
    stats[1].metric("Cached generations", len(generation_files))
    stats[2].metric("Saved benchmark files", len(results))
    stats[3].metric("Saved single runs", len(runs))
    st.code(str(cache_dir), language=None)
    if "single_result" in st.session_state:
        detailed = st.session_state["single_result"][0]
        st.subheader("Latest run telemetry")
        st.json(detailed.cache_stats or {"backend": "local", "backend_attempts": 0})
    st.caption(
        "Cache keys include model, prompt, schema, image bytes, embedding dimension, and full prediction payload."
    )
