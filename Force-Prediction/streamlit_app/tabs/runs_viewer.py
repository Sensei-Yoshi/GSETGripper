"""Saved suite, benchmark, and single-run inspection."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from modules.benchmarking import (
    BenchmarkEvaluation,
    BenchmarkPredictionBatch,
    evaluation_readiness,
    list_batch_evaluations,
    list_prediction_batches,
)
from modules.experiments import experiment_eligibility
from modules.reporting import (
    calibration_figure,
    common_intersection_artifacts,
    comparison_rows,
    force_error_statistics_rows,
    individual_calibration_figure,
    metrics_rows,
    suite_force_by_object_figure,
    suite_percentage_error_figure,
)
from modules.suites import (
    PRIMARY_EXPERIMENTS,
    create_suite,
    evaluate_suite,
    is_legacy_suite,
    list_suites,
    load_suite,
    run_suite_predictions,
    suite_evaluation_artifacts,
    suite_experiments,
    suite_prediction_batches,
)
from streamlit_app.benchmark_inspector import render_benchmark_object_inspector
from streamlit_app.context import AppContext
from streamlit_app.tabs.data_viewer import pipeline_run_inspector


def _suite_label(manifest: dict) -> str:
    backend = manifest["backend"]
    return (
        f"{manifest['created_at'][:19]} | {manifest['status'].replace('_', ' ').title()} | "
        f"{backend} | {manifest['suite_id']}"
    )


def _render_suite_provenance(manifest: dict) -> None:
    snapshot = manifest["definition_snapshot"]
    st.write(f"**Suite ID:** `{manifest['suite_id']}`")
    st.write(f"**Status:** {manifest['status'].replace('_', ' ').title()}")
    st.write(f"**Backend:** {manifest['backend']}")
    st.write(f"**Prompt bundle SHA-256:** `{snapshot.get('prompt_bundle_sha256', 'unknown')}`")
    st.write(
        f"**Definition version:** {snapshot.get('experiment_definition_version', 'unknown')}"
    )
    st.write(
        "**Active grippers:** "
        + ", ".join(snapshot.get("active_grippers", ("gecko", "silicone")))
    )
    st.write(f"**VLM:** {snapshot.get('models', {}).get('vlm', 'unknown')}")
    st.write(f"**Embedding:** {snapshot.get('models', {}).get('embedding', 'unknown')}")
    with st.expander("Experiment definitions and configuration lock"):
        st.json(
            {
                "experiments": snapshot.get("experiment_definitions", {}),
                "split": snapshot.get("split", {}),
                "split_sha256": snapshot.get("split_sha256"),
                "retrieval": snapshot.get("retrieval", {}),
                "inputs": snapshot.get("inputs", {}),
            },
            expanded=True,
        )
    with st.expander("Exact prompts and gripper context"):
        st.json(manifest.get("prompt_context", {}), expanded=True)


def _render_comparison(
    context: AppContext,
    manifest: dict,
    artifacts: dict[str, dict],
    suite_evaluation: dict | None = None,
) -> None:
    completed = [name.upper() for name in artifacts]
    st.caption(f"Available evaluation artifacts: {', '.join(completed) or 'none'}")
    if not artifacts:
        st.info(
            "No suite predictions have evaluation-ready truth yet. Generated predictions "
            "remain saved and can be evaluated later."
        )
        return

    counts = {
        experiment.upper(): len(artifact.get("rows", []))
        for experiment, artifact in artifacts.items()
    }
    st.write("**Per-experiment evaluated samples:**", counts)
    comparable, common_ids = common_intersection_artifacts(artifacts, context.config)
    if not common_ids:
        st.info(
            "A cross-experiment comparison needs at least one object evaluated in every "
            "experiment in this suite. Individual metrics remain available below."
        )
        st.subheader("Per-experiment metrics")
        st.dataframe(
            pd.DataFrame(metrics_rows(artifacts)),
            hide_index=True,
            width="stretch",
        )
        return
    st.caption(
        f"Cross-experiment comparison uses the common intersection of {len(common_ids)} objects."
    )

    st.subheader("Force predictions by object")
    force_figure = suite_force_by_object_figure(
        comparable,
        image_root=context.config.root,
    )
    st.pyplot(force_figure, width="stretch")
    import matplotlib.pyplot as plt

    plt.close(force_figure)

    st.subheader("Signed percentage error by object")
    percentage_figure = suite_percentage_error_figure(
        comparable,
        image_root=context.config.root,
    )
    st.pyplot(percentage_figure, width="stretch")
    plt.close(percentage_figure)

    st.subheader("True-vs-predicted calibration grid")
    figure = calibration_figure(comparable)
    st.pyplot(figure, width="stretch")
    plt.close(figure)

    st.subheader("Force-error statistics")
    statistics = pd.DataFrame(force_error_statistics_rows(comparable)).rename(
        columns={
            "experiment": "Experiment",
            "scope": "Scope",
            "n": "n",
            "mae_n": "MAE (N)",
            "rmse_n": "RMSE (N)",
            "residual_std_n": "Residual SD (N)",
            "overprediction_count": "Over count",
            "underprediction_count": "Under count",
            "exact_prediction_count": "Exact count",
            "average_overprediction_n": "Avg. over (N)",
            "average_underprediction_n": "Avg. under (N)",
        }
    )
    numeric_columns = [
        "MAE (N)",
        "RMSE (N)",
        "Residual SD (N)",
        "Avg. over (N)",
        "Avg. under (N)",
    ]
    st.dataframe(
        statistics.style.format(
            {column: "{:.3f}" for column in numeric_columns},
            na_rep="N/A",
        ),
        hide_index=True,
        width="stretch",
    )

    with st.expander("Comprehensive evaluation metrics"):
        st.caption("All available rows by experiment")
        st.dataframe(
            pd.DataFrame(metrics_rows(artifacts)),
            hide_index=True,
            width="stretch",
        )
        st.caption("Common-object intersection")
        st.dataframe(
            pd.DataFrame(metrics_rows(comparable)),
            hide_index=True,
            width="stretch",
        )

    st.subheader("Object detail")
    long_frame = pd.DataFrame(comparison_rows(comparable))
    selected_object = st.selectbox(
        "Object",
        sorted(long_frame["object_id"].unique()),
        key="suite_object_detail",
    )
    detail_columns = [
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
    st.dataframe(
        long_frame[long_frame["object_id"] == selected_object][detail_columns],
        hide_index=True,
        width="stretch",
    )
    exports = (suite_evaluation or {}).get("exports") or manifest.get("exports")
    if exports:
        st.caption("Saved comparison exports")
        st.json(exports, expanded=False)


def _suite_comparison(context: AppContext) -> None:
    cfg = context.config
    st.subheader("Experiment suite comparison")
    st.caption(
        "Prediction generation and evaluation are separate. Completed prediction batches "
        "remain immutable while missing conditions can become ready later."
    )
    confirmed = st.checkbox(
        "Confirm Gemini cost",
        value=False,
        help="A full suite makes one force-prediction call per eligible object and condition.",
        key="suite_cost_confirmation",
    )

    suites = list_suites(cfg)
    options = {_suite_label(item): item for item in suites}
    selected_label = st.selectbox(
        "Saved suite",
        ["Start a new suite", *options],
        key="suite_selector",
    )
    selected = options.get(selected_label)
    if selected is not None and selected.get("manifest_path"):
        selected = load_suite(selected["manifest_path"])

    # A new suite runs a chosen subset; a resumed suite is fixed to its own set.
    if selected is None:
        chosen = st.multiselect(
            "Experiments to run",
            list(PRIMARY_EXPERIMENTS),
            default=["e1", "e2", "e3", "e4"],
            format_func=str.upper,
            key="suite_experiment_choice",
            help=(
                "Surface-area E5 is optional. The current test set has no held-out "
                "contact-varied objects, so it cannot separate E5 from E4."
            ),
        )
        active_experiments = tuple(
            name for name in PRIMARY_EXPERIMENTS if name in set(chosen)
        )
    else:
        active_experiments = suite_experiments(selected)

    legacy_suite = selected is not None and is_legacy_suite(selected)
    if legacy_suite:
        st.info(
            "This definition-v12 suite is displayed with its E3-E6 IDs translated to "
            "E2-E5. It remains immutable; start a new suite to run the contiguous IDs."
        )

    pending_counts: dict[str, int] = {}
    for experiment in active_experiments:
        completed = (
            selected is not None
            and selected["runs"][experiment].get("status") == "completed"
        )
        pending_counts[experiment] = (
            0
            if completed or legacy_suite
            else len(experiment_eligibility(context.dataset, cfg, experiment).query_ids)
        )
    call_count = sum(pending_counts.values())
    st.caption(
        f"Up to {call_count} currently ready prediction calls; exact cached requests are reused."
    )

    action_columns = st.columns(2)
    run_clicked = action_columns[0].button(
        "Run/Resume suite predictions",
        type="primary",
        disabled=(
            not confirmed
            or call_count == 0
            or not active_experiments
            or legacy_suite
        ),
        key="run_primary_suite",
        width="stretch",
    )

    if run_clicked:
        if selected is None:
            selected = create_suite(cfg.model_copy(deep=True), active_experiments)
        offsets: dict[str, int] = {}
        cumulative = 0
        for experiment, count in pending_counts.items():
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

        with st.spinner("Generating provenance-locked suite predictions…"):
            selected = run_suite_predictions(cfg, selected, progress=progress)
        progress_bar.empty()
        status.empty()
        st.success("Currently ready suite predictions were saved.")

    can_evaluate = False
    if selected is not None and not legacy_suite:
        can_evaluate = any(
            evaluation_readiness(cfg, batch).eligible_ids
            for batch in suite_prediction_batches(cfg, selected).values()
        )
    evaluate_clicked = action_columns[1].button(
        "Evaluate suite & generate comparison",
        disabled=not can_evaluate or legacy_suite,
        key="evaluate_primary_suite",
        width="stretch",
    )

    if evaluate_clicked and selected is not None:
        with st.spinner("Evaluating saved suite predictions and generating plots…"):
            selected = evaluate_suite(cfg, selected)
        st.success("Suite evaluation was saved without model calls.")

    if selected is None:
        return

    _render_suite_provenance(selected)
    states = pd.DataFrame(
        [
            {
                "experiment": experiment.upper(),
                "status": state.get("status"),
                "prediction_count": state.get("query_count", 0),
                "prediction_batch_id": state.get("prediction_batch_id"),
                "reason": state.get("reason"),
            }
            for experiment, state in selected.get("runs", {}).items()
        ]
    )
    st.subheader("Prediction status")
    st.dataframe(states, hide_index=True, width="stretch")

    evaluations = list(reversed(selected.get("evaluations", [])))
    if evaluations:
        labels = {
            f"{item['created_at'][:19]} | {item['evaluation_id']}": item
            for item in evaluations
        }
        selected_evaluation = labels[
            st.selectbox(
                "Suite evaluation version",
                list(labels),
                key="suite_evaluation_selector",
            )
        ]
        artifacts = suite_evaluation_artifacts(
            cfg,
            selected,
            selected_evaluation,
        )
        _render_comparison(context, selected, artifacts, selected_evaluation)
    else:
        _render_comparison(context, selected, {})


def _batch_label(batch: BenchmarkPredictionBatch) -> str:
    targets = "+".join(batch.metadata["active_grippers"])
    return (
        f"{batch.display_name} | {batch.experiment_id.upper()} | "
        f"{targets} | {len(batch.rows)} rows"
    )


def _scalar_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {key: value for key, value in row.items() if key != "pipeline_result"}
            for row in rows
        ]
    )


def _render_metrics(evaluation: BenchmarkEvaluation) -> None:
    coverage = evaluation.metadata["coverage"]
    force = evaluation.metrics["force"]["overall"]
    selection = evaluation.metrics["selection"]
    columns = st.columns(4)
    columns[0].metric("Coverage", f"{coverage['evaluated']}/{coverage['predicted']}")
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


def _render_prediction_batch(
    context: AppContext,
    batch: BenchmarkPredictionBatch,
) -> None:
    evaluations = list_batch_evaluations(context.config, batch.batch_id)
    if evaluations:
        labels = {
            f"{item.metadata['created_at'][:19]} | "
            f"{item.metadata['coverage']['evaluated']}/{item.metadata['coverage']['predicted']}": item
            for item in evaluations
        }
        selected_evaluation = labels[
            st.selectbox(
                "Evaluation version",
                list(labels),
                key="benchmark_evaluation_viewer",
            )
        ]
    else:
        selected_evaluation = None

    (
        summary_tab,
        inspector_tab,
        plots_tab,
        evaluated_tab,
        predictions_tab,
        provenance_tab,
    ) = st.tabs(
        [
            "Summary",
            "Object Inspector",
            "Plots",
            "Evaluated Rows",
            "All Predictions",
            "Provenance",
        ]
    )
    with summary_tab:
        st.write(f"**Benchmark name:** {batch.display_name}")
        st.write(f"**Prediction batch:** `{batch.batch_id}`")
        st.write(
            f"**Experiment/method:** {batch.experiment_id.upper()} / "
            f"{batch.metadata['experiment_method']}"
        )
        st.write(f"**Predictions:** {len(batch.rows)}")
        st.write(f"**Active grippers:** {', '.join(batch.metadata['active_grippers'])}")
        st.write("**VLM roughness representation:** Continuous index")
        if selected_evaluation is None:
            readiness = evaluation_readiness(context.config, batch)
            st.info(
                f"No saved evaluation. {len(readiness.eligible_ids)} of {len(batch.rows)} "
                "rows currently have evaluation-ready truth."
            )
        else:
            _render_metrics(selected_evaluation)
            if selected_evaluation.metadata.get("skipped_truth"):
                with st.expander("Unevaluated predictions"):
                    st.json(selected_evaluation.metadata["skipped_truth"], expanded=False)

    with inspector_tab:
        render_benchmark_object_inspector(
            context.config,
            batch,
            selected_evaluation,
        )

    with plots_tab:
        if selected_evaluation is None:
            st.info("Plots become available after this batch is evaluated.")
        else:
            png_relative = selected_evaluation.metadata.get("exports", {}).get("png")
            png_path = context.config.root / png_relative if png_relative else None
            if png_path is not None and png_path.is_file():
                st.image(str(png_path), width="stretch")
            else:
                figure = individual_calibration_figure(
                    selected_evaluation.to_artifact(),
                    image_root=context.config.root,
                )
                st.pyplot(figure, width="stretch")
                import matplotlib.pyplot as plt

                plt.close(figure)

    with evaluated_tab:
        if selected_evaluation is None:
            st.info("No evaluated rows yet.")
        else:
            st.dataframe(
                _scalar_frame(selected_evaluation.rows),
                hide_index=True,
                width="stretch",
            )

    with predictions_tab:
        st.dataframe(_scalar_frame(batch.rows), hide_index=True, width="stretch")

    with provenance_tab:
        st.json(batch.metadata, expanded=True)
        if selected_evaluation is not None:
            st.subheader("Evaluation provenance")
            st.json(selected_evaluation.metadata, expanded=True)


def _benchmark_runs(context: AppContext) -> None:
    batches = list_prediction_batches(context.config)
    if not batches:
        st.info("No saved benchmark predictions. Use Benchmark first.")
        return
    options = {
        _batch_label(batch): batch for batch in batches
    }
    selected_label = st.selectbox(
        "Benchmark artifact",
        list(options),
        key="benchmark_run_viewer",
    )
    _render_prediction_batch(context, options[selected_label])


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
