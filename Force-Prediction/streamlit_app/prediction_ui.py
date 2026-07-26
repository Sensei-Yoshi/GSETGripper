"""Shared rendering helpers for pipeline predictions."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from modules.config import Config
from modules.contracts import Gripper
from modules.experiments import EXPERIMENT_CATALOG, experiment_display_name
from modules.pipeline import PipelineRunResult
from modules.retrieval import normalized_weights


def format_force(value: float, *, signed: bool = False) -> str:
    spec = "+.6g" if signed else ".6g"
    return f"{format(value, spec)} N"


def format_experiment(value: str) -> str:
    return experiment_display_name(value) if value.lower() in EXPERIMENT_CATALOG else value

def truth_for_display(obj) -> dict:  # noqa: ANN001
    optimal, _ = obj.optimal_grippers()
    if len(optimal) != 1:
        raise ValueError(f"validation object {obj.object_id!r} does not have a strict winner")
    return {
        "true_gecko_force_n": obj.gecko.min_force_n if obj.gecko else None,
        "true_silicone_force_n": obj.silicone.min_force_n if obj.silicone else None,
        "true_selection": next(iter(optimal)).value,
    }


def truth_payload(obj) -> dict:  # noqa: ANN001
    return {
        **truth_for_display(obj),
        "object_id": obj.object_id,
        "gecko_feasible": obj.gecko.feasible if obj.gecko else None,
        "silicone_feasible": obj.silicone.feasible if obj.silicone else None,
    }


def paired_retrieval_table(result: PipelineRunResult) -> pd.DataFrame:
    rows = []
    for item in result.retrieved_objects:
        sim = item.similarity
        row = {
                "rank": item.rank,
                "object": item.object_id.replace("_", " "),
                "score": item.score,
                "semantic": sim.semantic,
                "gecko_force_n": item.gecko_min_force_n,
                "gecko_feasible": item.gecko_feasible,
                "silicone_force_n": item.silicone_min_force_n,
                "silicone_feasible": item.silicone_feasible,
            }
        if result.retrieval_mode == "hybrid":
            row.update(
                mass=sim.mass,
                roughness=sim.roughness,
                contact=sim.contact,
                mass_g=item.mass_g,
                roughness_class=item.roughness_class,
                contact_fraction=item.projected_contact_fraction,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def render_formula(cfg: Config) -> None:
    weights = normalized_weights(cfg)
    st.markdown('<div class="formula"><b>Hybrid similarity</b></div>', unsafe_allow_html=True)
    st.latex(
        r"S_i=w_s\cos(e_q,e_i)+w_m e^{-|\ln m_q-\ln m_i|/\sigma_m}"
        r"+w_r(1-|r_q-r_i|/4)+w_a e^{-|a_q-a_i|/\sigma_a}"
    )
    st.caption(
        "Normalized weights: "
        + " | ".join(f"{name} {value:.2f}" for name, value in weights.items())
    )


def render_semantic_formula() -> None:
    st.markdown(
        '<div class="formula"><b>Semantic-only similarity</b></div>',
        unsafe_allow_html=True,
    )
    st.latex(r"S_{E3}(q,i)=\cos(e_q,e_i)")
    st.caption("Mass, roughness, and projected contact do not enter E3 retrieval.")


def render_prediction(
    detailed: PipelineRunResult,
    truth,
    *,
    counterfactual: bool,
    baseline: PipelineRunResult | None,
    cfg: Config,
    experiment: str | None = None,
) -> None:
    result = detailed.selection
    metric_cols = st.columns(4)
    selected_label = result.desired_gripper.title()
    if result.prediction_tie:
        selected_label += " (tie-break)"
    metric_cols[0].metric("Selected gripper", selected_label)
    metric_cols[1].metric(
        "Selected force",
        format_force(result.predicted_normal_force_n)
        if result.predicted_normal_force_n is not None
        else "None",
    )
    metric_cols[2].metric(
        "VLM recommendation",
        result.model_recommended_gripper.title()
        if result.model_recommended_gripper is not None
        else "Not applicable",
    )
    active_experiment = experiment or st.session_state.get("last_experiment", "e4")
    metric_cols[3].metric("Experiment", format_experiment(active_experiment))

    if result.recommendation_agrees_with_selector is False:
        st.warning(
            "The VLM recommendation disagreed with the authoritative Python selector. "
            "The displayed command uses the lowest feasible predicted force."
        )

    if result.prediction_tie:
        st.warning(
            "Both grippers have the same predicted command force. Selection used "
            f"{result.tie_break_reason or 'the deterministic fallback rule'}."
        )

    if counterfactual:
        st.markdown(
            '<p class="status-warn">Counterfactual mode: source labels are not scored.</p>',
            unsafe_allow_html=True,
        )
        if baseline is not None:
            delta = (
                (result.predicted_normal_force_n or 0.0)
                - (baseline.selection.predicted_normal_force_n or 0.0)
            )
            st.metric("Selected-force change from baseline", format_force(delta, signed=True))
    else:
        truth_values = truth_for_display(truth)
        predicted_correct = (
            result.desired_gripper in {g.value for g in truth.optimal_grippers()[0]}
        )
        st.markdown(
            f'<p class="status-ok">Leave-one-out truth: {truth_values["true_selection"]}; '
            f'prediction {"correct" if predicted_correct else "incorrect"}.</p>',
            unsafe_allow_html=True,
        )
        ground_truth = st.columns(3)
        ground_truth[0].metric(
            "True gecko force",
            format_force(truth_values["true_gecko_force_n"])
            if truth_values["true_gecko_force_n"] is not None
            else "Infeasible",
        )
        ground_truth[1].metric(
            "True silicone force",
            format_force(truth_values["true_silicone_force_n"])
            if truth_values["true_silicone_force_n"] is not None
            else "Infeasible",
        )
        ground_truth[2].metric("True winning gripper", truth_values["true_selection"].title())

    pred_cols = st.columns(2)
    for column, gripper in zip(pred_cols, ("gecko", "silicone"), strict=True):
        pred = result.candidate_predictions[gripper]
        with column:
            st.subheader(gripper.title())
            truth_force = None
            if not counterfactual:
                truth_record = truth.get(Gripper(gripper))
                truth_force = truth_record.min_force_n if truth_record else None
            st.metric(
                "Predicted force",
                format_force(pred.predicted_normal_force_n),
                delta=(
                    f"{format_force(pred.predicted_normal_force_n - truth_force, signed=True)} error"
                    if truth_force is not None
                    else None
                ),
                delta_color="inverse",
            )
            physics = detailed.physics_estimates.get(gripper)
            if physics:
                raw_force = physics.get("raw_force_n")
                raw_label = format_force(raw_force) if raw_force is not None else "infeasible"
                st.caption(f"Physics model: continuous estimate {raw_label}")
            else:
                st.caption("Physics model: not used by this experiment")
            st.write(pred.reasoning_trace or "No reasoning trace for this experiment.")
            if pred.evidence_used:
                st.markdown("**Evidence used**")
                for item in pred.evidence_used:
                    st.markdown(f"- {item}")
            if pred.calculation_summary:
                st.markdown(f"**Calculation:** {pred.calculation_summary}")
            if pred.assumptions_and_uncertainty:
                st.markdown("**Assumptions and uncertainty**")
                for item in pred.assumptions_and_uncertainty:
                    st.markdown(f"- {item}")

    if result.model_recommendation_summary:
        st.subheader("VLM comparison rationale")
        st.write(result.model_recommendation_summary)
        for item in result.model_comparison_evidence:
            st.markdown(f"- {item}")

    st.subheader(f"Top {cfg.retrieval.k} reference matches")
    if detailed.retrieved_objects:
        retrieval_label = "E3" if detailed.retrieval_mode == "semantic_only" else "E4"
        if cfg.models.dry_run:
            st.caption(
                f"{retrieval_label} retrieves each object once with both gripper outcomes. Offline mode "
                "uses the deterministic paired-neighbor stand-in and makes no VLM request."
            )
        else:
            st.caption(
                f"{retrieval_label} retrieves each object once. Both gripper outcomes come from the same "
                "paired experience and are sent together in one VLM request."
            )
        st.dataframe(
            paired_retrieval_table(detailed),
            hide_index=True,
            width="stretch",
            column_config={key: st.column_config.NumberColumn(format="%.3f") for key in (
                "score", "semantic", "mass", "roughness", "contact"
            )},
        )
    else:
        st.info("This experiment does not use retrieval.")
    if detailed.retrieval_mode == "semantic_only":
        render_semantic_formula()
    elif detailed.retrieval_mode == "hybrid":
        render_formula(cfg)
