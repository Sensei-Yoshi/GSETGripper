"""Leave-one-object-out benchmark tab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from modules.config import EXPERIMENT_IDS
from modules.experiments import experiment_eligibility
from modules.expforce import run_benchmark, save_benchmark
from streamlit_app.context import AppContext
from streamlit_app.prediction_ui import format_experiment


def render(context: AppContext) -> None:
    base_cfg = context.config
    left, right = st.columns([0.28, 0.72], gap="large")
    with left:
        st.subheader("Benchmark run")
        experiment = st.selectbox(
            "Experiment",
            list(EXPERIMENT_IDS),
            index=3,
            format_func=format_experiment,
            key="benchmark_experiment",
        )
        eligibility = experiment_eligibility(context.dataset, base_cfg, experiment)
        object_count = len(eligibility.benchmark_ids)
        st.caption(
            f"{object_count} eligible of {len(context.dataset.objects)} objects for "
            f"{experiment.upper()}."
        )
        if eligibility.skipped_benchmarks:
            with st.expander("Skipped objects and reasons"):
                st.json(eligibility.skipped_benchmarks, expanded=False)
        run = st.button(
            f"Run {object_count}-object leave-one-out benchmark",
            type="primary",
            width="stretch",
            disabled=object_count == 0,
        )
        st.caption("Results are synthetic pipeline diagnostics, not real-world performance claims.")

    if run:
        cfg = base_cfg.model_copy(deep=True)
        progress_bar = right.progress(0.0)
        status = right.empty()

        def progress(done: int, total: int, name: str) -> None:
            progress_bar.progress(done / total)
            status.caption(f"{done}/{total}: {name.replace('_', ' ')}")

        with right, st.spinner("Evaluating all objects with leave-one-out training..."):
            benchmark = run_benchmark(cfg, experiment, progress=progress)
            paths = save_benchmark(cfg, benchmark)
            st.session_state["benchmark_result"] = (benchmark, paths)
        progress_bar.empty()
        status.empty()

    with right:
        st.subheader("Leave-one-out results")
        if "benchmark_result" not in st.session_state:
            if object_count:
                st.info(
                    f"Run {object_count} eligible objects with each query excluded from its "
                    "own reference set."
                )
            else:
                st.info(f"No objects currently satisfy {experiment.upper()} benchmark truth and inputs.")
            return
        benchmark, paths = st.session_state["benchmark_result"]
        force = benchmark.metrics["force"]["overall"]
        selection = benchmark.metrics["selection"]
        recommendation = benchmark.metrics["model_recommendation"]
        metrics = st.columns(5)
        metrics[0].metric("Force MAE", f"{force.get('mae', float('nan')):.3f} N")
        metrics[1].metric("Force RMSE", f"{force.get('rmse', float('nan')):.3f} N")
        metrics[2].metric("Selection accuracy", f"{selection['accuracy']:.1%}")
        metrics[3].metric("Mean regret", f"{selection['mean_regret_n']:.3f} N")
        metrics[4].metric(
            "VLM recommendation accuracy",
            f"{recommendation['accuracy']:.1%}" if recommendation["n"] else "N/A",
        )
        result_frame = pd.DataFrame(benchmark.rows)
        plot_frame = pd.concat(
            [
                result_frame[["object_id", "true_gecko_force_n", "pred_gecko_force_n"]]
                .rename(columns={"true_gecko_force_n": "true", "pred_gecko_force_n": "pred"})
                .assign(gripper="gecko"),
                result_frame[["object_id", "true_silicone_force_n", "pred_silicone_force_n"]]
                .rename(columns={"true_silicone_force_n": "true", "pred_silicone_force_n": "pred"})
                .assign(gripper="silicone"),
            ],
            ignore_index=True,
        )
        st.scatter_chart(plot_frame, x="true", y="pred", color="gripper", size=70)
        display_columns = [
            "object_id", "mass_g", "roughness_class", "projected_contact_fraction",
            "true_gecko_force_n", "pred_gecko_force_n", "true_silicone_force_n",
            "pred_silicone_force_n", "true_favored", "predicted_gripper",
            "selection_correct", "regret_n",
            "model_recommended_gripper", "recommendation_agrees_with_selector",
        ]
        st.dataframe(result_frame[display_columns], hide_index=True, width="stretch")
        st.caption(f"Saved: {paths[0].name} and {paths[1].name}")
