"""Shared force estimators and the authoritative deterministic selector.

E1, E2, and E4 use one joint VLM response for both grippers. E5 converts the
calibrated physics solve into the shared per-gripper contract; E6 adds its
residual in :mod:`force_prediction.experiments`. Python always makes the final
feasible minimum-force selection and records agreement with the VLM's explicit
recommendation.
"""

from __future__ import annotations

import numpy as np

from .config import Config
from .contracts import (
    Compatibility,
    Gripper,
    GripperChoice,
    JointGripperPrediction,
    PerGripperPrediction,
    Query,
    SelectionResult,
)
from .llm import get_client
from .physics import PhysicsEstimate
from .retrieval import RetrievedObjectExperience, normalized_weights

GRIPPERS = (Gripper.GECKO, Gripper.SILICONE)


def _force_constraints(cfg: Config) -> dict:
    return {
        "minimum_n": cfg.force.min_n,
        "maximum_n": cfg.force.limit_n,
        "continuous_command": True,
    }


def clamp_force(force: float, cfg: Config) -> float:
    """Keep a continuous force estimate within the hardware command range."""
    return min(cfg.force.limit_n, max(cfg.force.min_n, float(force)))


def _query_payload(query: Query, cfg: Config, *, include_measured: bool) -> dict:
    # Object IDs are operational metadata and may contain human-readable names;
    # excluding them keeps E1 genuinely image-only and avoids semantic leakage.
    payload: dict = {}
    if include_measured:
        payload.update(mass_g=query.mass_g, roughness_class=query.roughness_class)
        if cfg.inputs.use_projected_contact:
            payload["projected_contact_fraction"] = query.projected_contact_fraction
    return payload


def vlm_predict_joint(
    cfg: Config,
    query: Query,
    image_bgr: np.ndarray | None,
    retrieved: list[RetrievedObjectExperience],
    *,
    instruction: str,
    include_measured: bool,
    include_retrieval: bool,
) -> JointGripperPrediction:
    """Estimate both grippers and recommend one with exactly one force-generation call."""
    payload: dict = {
        "query": _query_payload(query, cfg, include_measured=include_measured),
        "force_constraints": _force_constraints(cfg),
    }
    if include_measured:
        payload["roughness_scale"] = cfg.roughness.labels
    if include_retrieval:
        payload["retrieved_objects"] = [
            item.to_payload(include_contact=cfg.inputs.use_projected_contact)
            for item in retrieved
        ]
        payload["retrieval_config"] = {
            "k": cfg.retrieval.k,
            "normalized_weights": normalized_weights(cfg),
            "sigma_mass": cfg.retrieval.sigma_mass,
            "sigma_contact": cfg.retrieval.sigma_contact,
        }

    if cfg.models.dry_run:
        return _joint_stub(cfg, query, retrieved if include_retrieval else [])

    raw = get_client(cfg).generate_json(
        system=cfg.prompts.prediction_system,
        instruction=instruction,
        schema=JointGripperPrediction,
        image_bgr=image_bgr,
        extra=payload,
    )
    response = JointGripperPrediction.model_validate(raw)
    response.gecko.candidate_gripper = Gripper.GECKO
    response.silicone.candidate_gripper = Gripper.SILICONE
    response.gecko.predicted_normal_force_n = clamp_force(
        response.gecko.predicted_normal_force_n, cfg
    )
    response.silicone.predicted_normal_force_n = clamp_force(
        response.silicone.predicted_normal_force_n, cfg
    )
    return response


def predictions_from_joint(
    response: JointGripperPrediction,
) -> dict[Gripper, PerGripperPrediction]:
    return {
        Gripper.GECKO: response.gecko,
        Gripper.SILICONE: response.silicone,
    }


def _joint_stub(
    cfg: Config,
    query: Query,
    retrieved: list[RetrievedObjectExperience],
) -> JointGripperPrediction:
    """Deterministic offline stand-in for plumbing tests; it is not an accuracy baseline."""
    if retrieved:
        predictions = {
            gripper: _paired_retrieval_average_predict(cfg, retrieved, gripper)
            for gripper in GRIPPERS
        }
    else:
        base = 0.002 * query.mass_g + 0.25
        gecko_scale = 0.88 if query.roughness_class <= 2 else 1.12
        silicone_scale = 1.08 if query.roughness_class <= 2 else 0.94
        predictions = {
            Gripper.GECKO: PerGripperPrediction(
                candidate_gripper=Gripper.GECKO,
                feasible=True,
                predicted_normal_force_n=clamp_force(base * gecko_scale, cfg),
                reasoning_trace="dry-run joint stub",
            ),
            Gripper.SILICONE: PerGripperPrediction(
                candidate_gripper=Gripper.SILICONE,
                feasible=True,
                predicted_normal_force_n=clamp_force(base * silicone_scale, cfg),
                reasoning_trace="dry-run joint stub",
            ),
        }
    recommendation = select(predictions).desired_gripper
    return JointGripperPrediction(
        gecko=predictions[Gripper.GECKO],
        silicone=predictions[Gripper.SILICONE],
        recommended_gripper=recommendation,
        recommendation_summary="dry-run recommendation from the joint stub",
    )


def _paired_retrieval_average_predict(
    cfg: Config,
    retrieved: list[RetrievedObjectExperience],
    gripper: Gripper,
) -> PerGripperPrediction:
    """Offline E4 stand-in using the same shared paired-object neighbor list."""
    force_field = (
        "gecko_min_force_n" if gripper is Gripper.GECKO else "silicone_min_force_n"
    )
    feasible_field = "gecko_feasible" if gripper is Gripper.GECKO else "silicone_feasible"
    usable = [
        item
        for item in retrieved
        if getattr(item, feasible_field) and getattr(item, force_field) is not None
    ]
    if not usable:
        return PerGripperPrediction(
            candidate_gripper=gripper,
            feasible=False,
            predicted_normal_force_n=cfg.force.limit_n,
            reasoning_trace="no feasible paired neighbors in dry-run stub",
        )
    weights = np.asarray([max(0.0, item.score) for item in usable], dtype=float)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    forces = np.asarray([getattr(item, force_field) for item in usable], dtype=float)
    estimate = float((weights * forces).sum() / weights.sum())
    return PerGripperPrediction(
        candidate_gripper=gripper,
        feasible=True,
        predicted_normal_force_n=clamp_force(estimate, cfg),
        reasoning_trace="dry-run estimate from shared paired-object neighbors",
    )


def physics_predict(
    cfg: Config,
    gripper: Gripper,
    physics_estimate: PhysicsEstimate,
) -> PerGripperPrediction:
    """Convert an E5 calibrated-physics solve into the common prediction contract."""
    return PerGripperPrediction(
        candidate_gripper=gripper,
        feasible=physics_estimate.feasible,
        predicted_normal_force_n=(
            physics_estimate.min_force_n
            if (physics_estimate.feasible and physics_estimate.min_force_n is not None)
            else cfg.force.limit_n
        ),
        reasoning_trace="calibrated reduced-order physics",
    )


def select(
    predictions: dict[Gripper, PerGripperPrediction],
    reasoning: str = "",
    *,
    model_recommended_gripper: GripperChoice | None = None,
    model_recommendation_summary: str | None = None,
) -> SelectionResult:
    """Choose the lowest-force feasible gripper; VLM recommendations are diagnostic only."""
    feasible = [prediction for prediction in predictions.values() if prediction.feasible]
    candidate_map = {gripper.value: prediction for gripper, prediction in predictions.items()}

    if not feasible:
        desired_gripper = "none"
        return SelectionResult(
            desired_gripper=desired_gripper,
            predicted_normal_force_n=None,
            candidate_predictions=candidate_map,
            reasoning_trace=reasoning or "no feasible gripper",
            model_recommended_gripper=model_recommended_gripper,
            model_recommendation_summary=model_recommendation_summary,
            recommendation_agrees_with_selector=(
                desired_gripper == model_recommended_gripper
                if model_recommended_gripper is not None
                else None
            ),
        )

    minimum = min(prediction.predicted_normal_force_n for prediction in feasible)
    tied = [prediction for prediction in feasible if prediction.predicted_normal_force_n == minimum]
    compatibility_rank = {
        Compatibility.HIGH: 3,
        Compatibility.MEDIUM: 2,
        Compatibility.LOW: 1,
        Compatibility.UNKNOWN: 0,
    }
    best = max(tied, key=lambda prediction: compatibility_rank[prediction.compatibility])
    tie_break_reason = None
    if len(tied) > 1:
        ranks = {compatibility_rank[prediction.compatibility] for prediction in tied}
        tie_break_reason = (
            "higher predicted material compatibility"
            if len(ranks) > 1
            else "stable gripper order because force and compatibility were equal"
        )
    desired_gripper = best.candidate_gripper.value
    return SelectionResult(
        desired_gripper=desired_gripper,
        predicted_normal_force_n=best.predicted_normal_force_n,
        candidate_predictions=candidate_map,
        reasoning_trace=(
            reasoning
            or (
                f"predicted-force tie resolved by {tie_break_reason}"
                if tie_break_reason
                else "lowest feasible predicted stationary-finger force"
            )
        ),
        prediction_tie=len(tied) > 1,
        tie_break_reason=tie_break_reason,
        model_recommended_gripper=model_recommended_gripper,
        model_recommendation_summary=model_recommendation_summary,
        recommendation_agrees_with_selector=(
            desired_gripper == model_recommended_gripper
            if model_recommended_gripper is not None
            else None
        ),
    )
