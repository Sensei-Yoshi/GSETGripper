"""Two-stage benchmark prediction and evaluation controls."""

from __future__ import annotations

import streamlit as st

from modules.benchmarking import (
    BenchmarkEvaluation,
    BenchmarkPredictionBatch,
    EvaluationProtocol,
    evaluation_readiness,
    generate_benchmark_predictions,
    get_or_create_benchmark_evaluation,
    list_batch_evaluations,
    list_prediction_batches,
    save_prediction_batch,
)
from modules.config import EXPERIMENT_IDS
from modules.experiments import experiment_eligibility
from streamlit_app.context import AppContext
from streamlit_app.prediction_ui import format_experiment


def _batch_label(batch: BenchmarkPredictionBatch) -> str:
    targets = "+".join(batch.metadata["active_grippers"])
    protocol = batch.metadata.get(
        "evaluation_protocol", EvaluationProtocol.NEW_SURFACE.value
    ).replace("_", " ")
    return (
        f"{batch.metadata['created_at'][:19]} | {protocol} | {targets} | "
        f"{len(batch.rows)} predictions"
    )


def _render_evaluation_summary(
    evaluation: BenchmarkEvaluation,
    *,
    reused: bool = False,
) -> None:
    coverage = evaluation.metadata["coverage"]
    force = evaluation.metrics["force"]["overall"]
    selection = evaluation.metrics["selection"]
    columns = st.columns(4)
    columns[0].metric(
        "Coverage",
        f"{coverage['evaluated']}/{coverage['predicted']}",
    )
    columns[1].metric(
        "Force MAE",
        f"{force['mae']:.3f} N" if force.get("n") else "N/A",
    )
    columns[2].metric(
        "Force RMSE",
        f"{force['rmse']:.3f} N" if force.get("n") else "N/A",
    )
    columns[3].metric(
        "Selection accuracy",
        f"{selection['accuracy']:.1%}" if selection.get("applicable") else "N/A",
    )
    if reused:
        st.info("Current truth matches an existing evaluation; no duplicate was created.")
    st.caption(
        "Detailed plots, evaluated rows, all predictions, and provenance are available "
        "in Runs Viewer → Benchmark Runs."
    )


def render(context: AppContext) -> None:
    cfg = context.config
    control_col, summary_col = st.columns([0.38, 0.62], gap="large")

    def clear_benchmark_summary() -> None:
        st.session_state.pop("benchmark_prediction_result", None)
        st.session_state.pop("benchmark_evaluation_result", None)

    def clear_evaluation_summary() -> None:
        st.session_state.pop("benchmark_evaluation_result", None)

    with control_col:
        st.subheader("1. Generate predictions")
        experiment = st.selectbox(
            "Experiment",
            list(EXPERIMENT_IDS),
            index=0,
            format_func=format_experiment,
            key="benchmark_experiment",
            on_change=clear_benchmark_summary,
        )
        generation = experiment_eligibility(context.dataset, cfg, experiment)
        protocol = st.selectbox(
            "Evaluation protocol",
            list(EvaluationProtocol),
            format_func=lambda value: (
                "New-surface generalization"
                if value is EvaluationProtocol.NEW_SURFACE
                else "Known-surface condition interpolation"
            ),
            key="benchmark_protocol",
            on_change=clear_benchmark_summary,
        )
        st.caption(
            f"{len(generation.query_ids)} query-ready of "
            f"{len(context.dataset.objects)} objects for {experiment.upper()}."
        )
        if generation.skipped_queries:
            with st.expander("Generation exclusions"):
                st.json(generation.skipped_queries, expanded=False)

        button_columns = st.columns(2)
        run_predictions = button_columns[0].button(
            "Run selected",
            type="primary",
            width="stretch",
            disabled=not generation.query_ids,
            key="run_benchmark_predictions",
        )
        run_both = button_columns[1].button(
            "Run both",
            width="stretch",
            disabled=not generation.query_ids,
            key="run_both_benchmark_protocols",
        )
        st.caption(
            "This calls Gemini and saves immutable predictions. Force labels are not required."
        )

    if run_predictions or run_both:
        progress_bar = summary_col.progress(0.0)
        status = summary_col.empty()
        with summary_col, st.spinner("Generating and saving prediction batch…"):
            st.session_state.pop("benchmark_evaluation_result", None)
            protocols = list(EvaluationProtocol) if run_both else [protocol]
            last_result = None
            for protocol_index, selected_protocol in enumerate(protocols):
                def progress(
                    done: int,
                    total: int,
                    object_id: str,
                    protocol_index: int = protocol_index,
                    selected_protocol: EvaluationProtocol = selected_protocol,
                ) -> None:
                    overall_total = total * len(protocols)
                    overall_done = protocol_index * total + done
                    progress_bar.progress(overall_done / overall_total)
                    status.caption(
                        f"{selected_protocol.value.replace('_', ' ')} — "
                        f"{done}/{total}: {object_id.replace('_', ' ')}"
                    )

                batch = generate_benchmark_predictions(
                    cfg,
                    experiment,
                    protocol=selected_protocol,
                    progress=progress,
                )
                paths = save_prediction_batch(cfg, batch)
                last_result = (batch, paths)
            st.session_state["benchmark_prediction_result"] = last_result
        progress_bar.empty()
        status.empty()

    batches = list_prediction_batches(cfg, experiment=experiment)
    with control_col:
        st.divider()
        st.subheader("2. Evaluate saved predictions")
        if not batches:
            st.info(f"No saved {experiment.upper()} prediction batches yet.")
            selected_batch = None
        else:
            labels = {_batch_label(batch): batch for batch in batches}
            selected_label = st.selectbox(
                "Prediction batch",
                list(labels),
                key="benchmark_prediction_batch",
                on_change=clear_evaluation_summary,
            )
            selected_batch = labels[selected_label]

        if selected_batch is not None:
            readiness = evaluation_readiness(cfg, selected_batch)
            st.caption(
                f"{len(readiness.eligible_ids)} of {len(selected_batch.rows)} predictions "
                "currently have evaluation-ready truth."
            )
            if readiness.skipped:
                with st.expander("Truth still needed"):
                    st.json(readiness.skipped, expanded=False)
            evaluate = st.button(
                "Evaluate & generate plots",
                width="stretch",
                disabled=not readiness.eligible_ids,
                key="evaluate_benchmark_predictions",
            )
            st.caption("Evaluation uses saved predictions and makes no model calls.")
        else:
            evaluate = False

    if evaluate and selected_batch is not None:
        with summary_col, st.spinner("Scoring current truth and generating plots…"):
            evaluation, paths, reused = get_or_create_benchmark_evaluation(
                cfg,
                selected_batch,
            )
            st.session_state["benchmark_evaluation_result"] = (
                evaluation,
                paths,
                reused,
            )

    with summary_col:
        st.subheader("Benchmark status")
        latest_state = st.session_state.get("benchmark_evaluation_result")
        if latest_state is not None:
            evaluation, paths, reused = latest_state
            _render_evaluation_summary(evaluation, reused=reused)
            if paths:
                st.caption(
                    "Saved: " + ", ".join(path.name for path in paths.values())
                )
        elif selected_batch is not None:
            evaluations = list_batch_evaluations(cfg, selected_batch.batch_id)
            if evaluations:
                _render_evaluation_summary(evaluations[0])
            else:
                st.success(
                    f"Prediction batch contains {len(selected_batch.rows)} immutable rows."
                )
                st.info(
                    "Add force/feasibility labels, then return here to evaluate without "
                    "rerunning Gemini."
                )
        else:
            st.info("Generate a prediction batch to begin.")

        st.caption(
            "Benchmark outputs are pipeline diagnostics and are not real-world performance claims."
        )
