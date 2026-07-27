"""Saved suite, benchmark, and single-run inspection."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from modules.experiments import experiment_eligibility
from modules.expforce import artifact_backend_label
from modules.reporting import (
    calibration_figure,
    common_intersection_artifacts,
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
    backend = manifest.get("backend") or f"Legacy {manifest.get('execution_mode', 'Unknown')}"
    return (
        f"{manifest['created_at'][:19]} | {manifest['status'].title()} | "
        f"{backend} | {manifest['suite_id']}"
    )


def _render_suite_provenance(manifest: dict) -> None:
    snapshot = manifest["snapshot"]
    st.write(f"**Suite ID:** `{manifest['suite_id']}`")
    st.write(f"**Status:** {manifest['status'].title()}")
    backend = manifest.get("backend") or f"Legacy {manifest.get('execution_mode', 'Unknown')}"
    st.write(f"**Backend:** {backend}")
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
    counts = {
        experiment.upper(): len(artifact.get("rows", []))
        for experiment, artifact in artifacts.items()
    }
    st.write("**Per-experiment eligible samples:**", counts)
    skipped = {
        experiment.upper(): state.get("reason")
        for experiment, state in manifest.get("runs", {}).items()
        if state.get("status") == "skipped"
    }
    if skipped:
        st.warning(f"Skipped conditions: {skipped}")
    if not artifacts:
        st.info("No experiment currently has eligible benchmark objects.")
        return

    st.subheader("Per-experiment metrics")
    st.dataframe(pd.DataFrame(metrics_rows(artifacts)), hide_index=True, width="stretch")

    comparable, common_ids = common_intersection_artifacts(artifacts, context.config)
    if not common_ids:
        st.info(
            "A cross-experiment comparison needs at least one object eligible for all E1–E4 "
            "conditions. Individual metrics remain available above."
        )
        return
    st.caption(
        f"Cross-experiment comparison uses the common intersection of {len(common_ids)} objects."
    )

    figure = calibration_figure(comparable)
    st.pyplot(figure, width="stretch")
    import matplotlib.pyplot as plt

    plt.close(figure)
    st.subheader("Common-intersection metrics")
    st.dataframe(pd.DataFrame(metrics_rows(comparable)), hide_index=True, width="stretch")

    st.subheader("Object detail")
    long_frame = pd.DataFrame(comparison_rows(comparable))
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
        exports = export_comparison(comparable, destination)
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
    controls = st.columns([0.25, 0.75])
    confirmed = controls[0].checkbox(
        "Confirm Gemini cost",
        value=False,
        help="A full suite can make up to 516 Gemini force-prediction calls.",
        key="suite_cost_confirmation",
    )
    eligible_calls = sum(
        len(experiment_eligibility(context.dataset, cfg, experiment).benchmark_ids)
        for experiment in ("e1", "e2", "e3", "e4")
    )
    controls[1].warning(
        f"Up to {eligible_calls} Gemini force-prediction calls for currently eligible rows; "
        "exact cached requests are reused."
    )

    suites = list_suites(cfg)
    options = {_suite_label(item): item for item in suites}
    selected_label = st.selectbox(
        "Saved suite",
        ["Start a new suite", *options],
        key="suite_selector",
    )
    selected = options.get(selected_label)
    resumable = selected is None or selected.get("schema_version") == 7
    if not resumable:
        st.error(
            "This legacy suite is available for inspection only. Start a new Gemini suite "
            "to run or resume work."
        )
    action_label = "Resume selected suite" if selected else "Run new E1–E4 suite"
    if st.button(
        action_label,
        type="primary",
        disabled=not confirmed or not resumable,
        key="run_primary_suite",
    ):
        run_cfg = cfg.model_copy(deep=True)
        if selected is None:
            selected = create_suite(run_cfg)
        counts = {
            experiment: len(
                selected["snapshot"]["eligibility"][experiment][
                    "eligible_benchmark_ids"
                ]
            )
            for experiment in ("e1", "e2", "e3", "e4")
        }
        offsets: dict[str, int] = {}
        cumulative = 0
        for experiment, count in counts.items():
            offsets[experiment] = cumulative
            cumulative += count
        total_calls = max(cumulative, 1)
        progress_bar = st.progress(0.0)
        status = st.empty()

        def progress(experiment: str, done: int, total: int, object_id: str) -> None:
            progress_bar.progress((offsets[experiment] + done) / total_calls)
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
            f"{artifact_backend_label(item)}"
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
        _benchmark_runs(context)
    else:
        _suite_comparison(context)
