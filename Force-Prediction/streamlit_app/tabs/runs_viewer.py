"""Saved suite, benchmark, and single-run inspection."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from modules.reporting import (
    calibration_figure,
    comparison_rows,
    export_comparison,
    metrics_rows,
)
from modules.suites import (
    create_suite,
    list_suites,
    load_suite,
    run_suite,
    suite_benchmarks,
    update_suite_exports,
)
from streamlit_app.context import AppContext
from streamlit_app.tabs.data_viewer import pipeline_run_inspector


def _suite_label(manifest: dict) -> str:
    return (
        f"{manifest['created_at'][:19]} | {manifest['status'].title()} | "
        f"{manifest['execution_mode']} | {manifest['suite_id']}"
    )


def _render_suite_provenance(manifest: dict) -> None:
    snapshot = manifest["snapshot"]
    st.write(f"**Suite ID:** `{manifest['suite_id']}`")
    st.write(f"**Status:** {manifest['status'].title()}")
    st.write(f"**Execution:** {manifest['execution_mode']}")
    st.write(f"**Dataset SHA-256:** `{snapshot['source_sha256']}`")
    st.write(f"**Prompt bundle SHA-256:** `{snapshot['prompt_bundle_sha256']}`")
    st.write(f"**Definition version:** {snapshot['experiment_definition_version']}")
    st.write(f"**VLM:** {snapshot['models']['vlm']}")
    st.write(f"**Embedding:** {snapshot['models']['embedding']}")
    with st.expander("Experiment definitions and retrieval snapshot"):
        st.json(
            {
                "experiments": snapshot["experiment_definitions"],
                "retrieval": snapshot["retrieval"],
                "inputs": snapshot["inputs"],
            },
            expanded=True,
        )
    with st.expander("Exact prompts and gripper context"):
        st.json(manifest.get("prompt_context", {}), expanded=True)


def _render_comparison(context: AppContext, manifest: dict) -> None:
    artifacts = suite_benchmarks(context.config, manifest)
    completed = [name.upper() for name in artifacts]
    st.caption(f"Available benchmark artifacts: {', '.join(completed) or 'none'}")
    if len(artifacts) < 4:
        st.info("Complete or resume all E1–E4 conditions to render the comparison.")
        return

    figure = calibration_figure(artifacts)
    st.pyplot(figure, width="stretch")
    import matplotlib.pyplot as plt

    plt.close(figure)
    st.subheader("Comparative metrics")
    st.dataframe(pd.DataFrame(metrics_rows(artifacts)), hide_index=True, width="stretch")

    st.subheader("Object detail")
    long_frame = pd.DataFrame(comparison_rows(artifacts))
    selected_object = st.selectbox(
        "Object",
        sorted(long_frame["object_id"].unique()),
        key="suite_object_detail",
    )
    detail = long_frame[long_frame["object_id"] == selected_object][
        [
            "experiment",
            "gripper",
            "true_force_n",
            "predicted_force_n",
            "signed_error_n",
            "absolute_error_n",
            "predicted_gripper",
            "selection_correct",
            "regret_n",
        ]
    ]
    st.dataframe(detail, hide_index=True, width="stretch")

    if st.button("Export PNG, SVG, and CSV", key="export_suite_comparison"):
        destination = (
            context.dataset.paths.suites
            / manifest["suite_id"]
            / "exports"
        )
        exports = export_comparison(artifacts, destination)
        relative = {
            name: str(Path(path).relative_to(context.config.root))
            for name, path in exports.items()
        }
        update_suite_exports(context.config, manifest, relative)
        st.success("Saved publication exports.")
        st.json(relative, expanded=True)
    elif manifest.get("exports"):
        st.caption("Saved exports")
        st.json(manifest["exports"], expanded=False)


def _suite_comparison(context: AppContext) -> None:
    cfg = context.config
    st.subheader("E1–E4 suite comparison")
    st.caption(
        "A suite snapshots one dataset, prompt bundle, model, retrieval configuration, "
        "and experiment-definition version. Completed conditions are resumable."
    )
    controls = st.columns([0.25, 0.25, 0.5])
    mode = controls[0].segmented_control(
        "Execution",
        ["Offline", "Live Gemini"],
        default="Offline",
        key="suite_execution_mode",
    )
    live = mode == "Live Gemini"
    confirmed = controls[1].checkbox(
        "Confirm live cost",
        value=False,
        disabled=not live,
        help="A full live suite can make up to 516 force-prediction calls.",
        key="suite_live_confirmation",
    )
    if live:
        controls[2].warning(
            "Maximum 516 force-prediction calls, plus descriptor/embedding preparation."
        )
    else:
        controls[2].info("Offline mode uses deterministic plumbing stubs, not accuracy results.")

    suites = list_suites(cfg)
    options = {_suite_label(item): item for item in suites}
    selected_label = st.selectbox(
        "Saved suite",
        ["Start a new suite", *options],
        key="suite_selector",
    )
    selected = options.get(selected_label)
    mode_matches = (
        selected is None or selected["execution_mode"] == mode
    )
    if not mode_matches:
        assert selected is not None
        st.error(
            f"This suite was created as {selected['execution_mode']}. Select that "
            "execution mode to resume it, or start a new suite."
        )
    action_label = "Resume selected suite" if selected else "Run new E1–E4 suite"
    if st.button(
        action_label,
        type="primary",
        disabled=(live and not confirmed) or not mode_matches,
        key="run_primary_suite",
    ):
        run_cfg = cfg.model_copy(deep=True)
        run_cfg.models.dry_run = not live
        if selected is None:
            selected = create_suite(run_cfg)
        progress_bar = st.progress(0.0)
        status = st.empty()

        def progress(experiment: str, done: int, total: int, object_id: str) -> None:
            experiment_index = ("e1", "e2", "e3", "e4").index(experiment)
            progress_bar.progress((experiment_index * total + done) / (4 * total))
            status.caption(
                f"{experiment.upper()} | {done}/{total} | {object_id.replace('_', ' ')}"
            )

        with st.spinner("Running provenance-locked experiment suite…"):
            selected = run_suite(run_cfg, selected, progress=progress)
        progress_bar.empty()
        status.empty()
        st.success("Suite completed and saved.")

    if selected is not None:
        path = selected.get("manifest_path")
        if path:
            selected = load_suite(path)
        _render_suite_provenance(selected)
        _render_comparison(context, selected)


def _benchmark_runs(context: AppContext) -> None:
    root = context.dataset.paths.results
    paths = sorted(root.glob("*.json"), reverse=True)
    if not paths:
        st.info("No saved benchmark runs. Use Benchmark or Suite Comparison first.")
        return
    artifacts = []
    for path in paths:
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        artifact["artifact_path"] = path
        artifacts.append(artifact)
    labels = {
        (
            f"{item['metadata']['created_at'][:19]} | "
            f"{item['metadata']['experiment'].upper()} | "
            f"{'Offline' if item['metadata']['dry_run'] else 'Live Gemini'}"
        ): item
        for item in artifacts
    }
    selected = labels[st.selectbox("Benchmark run", list(labels), key="benchmark_run_viewer")]
    metadata = selected["metadata"]
    st.write(f"**Artifact:** `{selected['artifact_path'].name}`")
    st.write(f"**Experiment/method:** {metadata['experiment'].upper()} / {metadata['experiment_method']}")
    st.write(f"**Definition version:** {metadata['experiment_definition_version']}")
    st.write(f"**Dataset SHA-256:** `{metadata['source_sha256']}`")
    st.write(f"**Retrieval mode:** {metadata.get('retrieval_mode') or 'none'}")
    with st.expander("Exact run provenance"):
        st.json(metadata, expanded=True)
    st.subheader("Metrics")
    st.json(selected["metrics"], expanded=True)
    st.subheader("Object rows")
    st.dataframe(pd.DataFrame(selected["rows"]), hide_index=True, width="stretch")


def render(context: AppContext) -> None:
    st.header("Runs Viewer")
    mode = st.segmented_control(
        "View",
        ["Suite Comparison", "Benchmark Runs", "Single Runs"],
        default="Suite Comparison",
        key="runs_viewer_mode",
    )
    if mode == "Single Runs":
        pipeline_run_inspector(context)
    elif mode == "Benchmark Runs":
        if context.dataset.capabilities.can_benchmark:
            _benchmark_runs(context)
        else:
            st.info(
                f"{context.dataset.display_name} has no paired force labels, so it has no "
                "benchmark workflow. Use Data Viewer or Data Preparation for this dataset."
            )
    else:
        if context.dataset.capabilities.can_benchmark:
            _suite_comparison(context)
        else:
            st.info(
                f"{context.dataset.display_name} has no paired force labels, so experiment "
                "suites are unavailable. Use Data Viewer or Data Preparation instead."
            )
