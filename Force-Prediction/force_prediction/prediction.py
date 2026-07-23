"""Force estimators plus the deterministic selector.

Three estimators, all returning the same `PerGripperPrediction` contract so the
pipeline can swap them by experiment toggle without branching logic elsewhere:

  vlm_predict_gripper       — Gemini structured output (E1/E2/E3); a
                              physics/retrieval-grounded stub under dry_run.
  vlm_predict_paired        — one joint Gemini response for both E5 grippers.
  retrieval_average_predict — similarity-weighted average of top-k forces (E3b).
  physics_predict           — straight from the calibrated physics solver (E4).

`select` performs the final feasible arg-min in Python — never a second LLM call.
"""

from __future__ import annotations

import math

import numpy as np

from .config import Config
from .contracts import (
    CandidateQuery,
    Compatibility,
    Gripper,
    PairedGripperPrediction,
    PerGripperPrediction,
    Query,
    SelectionResult,
)
from .llm import get_client
from .physics import PhysicsEstimate
from .retrieval import RetrievedExperience, RetrievedObjectExperience, normalized_weights


def _force_constraints(cfg: Config) -> dict:
    return {
        "minimum_n": cfg.force.min_n,
        "maximum_n": cfg.force.limit_n,
        "resolution_n": cfg.force.increment_n,
    }


def _quantize(force: float, cfg: Config) -> float:
    inc = cfg.force.increment_n
    return round(min(cfg.force.limit_n, max(cfg.force.min_n, math.ceil(force / inc) * inc)), 6)


# --------------------------------------------------------------------------- #
# VLM estimator
# --------------------------------------------------------------------------- #
def vlm_predict_gripper(
    cfg: Config,
    query: CandidateQuery,
    image_bgr: np.ndarray | None,
    retrieved: list[RetrievedExperience],
    physics_estimate: PhysicsEstimate | None,
    include_paired: bool,
    include_measured: bool = True,
) -> PerGripperPrediction:
    # E1 (vision-only) hides measured properties + physics so the VLM must
    # estimate them from the image; other conditions reveal them.
    query_block: dict = {
        "object_id": query.object_id,
        "candidate_gripper": query.candidate_gripper.value,
    }
    if include_measured:
        query_block.update(
            mass_g=query.mass_g,
            roughness_class=query.roughness_class,
            projected_contact_fraction=query.projected_contact_fraction,
        )
    payload = {
        "query": query_block,
        "roughness_scale": cfg.roughness.labels,
        "retrieved_experiences": [r.to_payload(include_paired) for r in retrieved],
        "retrieval_config": {
            "normalized_weights": normalized_weights(cfg),
            "sigma_mass": cfg.retrieval.sigma_mass,
            "sigma_contact": cfg.retrieval.sigma_contact,
        },
        "force_constraints": _force_constraints(cfg),
    }
    if cfg.models.dry_run:
        return _stub_prediction(cfg, query, retrieved, physics_estimate)

    instruction = cfg.prompts.per_gripper_instruction.format(
        candidate_gripper=query.candidate_gripper.value
    )
    raw = get_client(cfg).generate_json(
        system=cfg.prompts.system,
        instruction=instruction,
        schema=PerGripperPrediction,
        image_bgr=image_bgr,
        extra=payload,
    )
    pred = PerGripperPrediction.model_validate(raw)
    pred.candidate_gripper = query.candidate_gripper  # trust our binding, not the model's
    pred.predicted_normal_force_n = _quantize(pred.predicted_normal_force_n, cfg)
    return pred


def vlm_predict_paired(
    cfg: Config,
    query: Query,
    image_bgr: np.ndarray | None,
    retrieved: list[RetrievedObjectExperience],
    include_measured: bool = True,
) -> dict[Gripper, PerGripperPrediction]:
    """Predict both grippers from one paired-object context and one VLM call."""
    query_block: dict = {"object_id": query.object_id}
    if include_measured:
        query_block.update(
            mass_g=query.mass_g,
            roughness_class=query.roughness_class,
            projected_contact_fraction=query.projected_contact_fraction,
        )
    payload = {
        "query": query_block,
        "roughness_scale": cfg.roughness.labels,
        "retrieved_objects": [item.to_payload() for item in retrieved],
        "retrieval_config": {
            "normalized_weights": normalized_weights(cfg),
            "sigma_mass": cfg.retrieval.sigma_mass,
            "sigma_contact": cfg.retrieval.sigma_contact,
        },
        "force_constraints": _force_constraints(cfg),
    }
    if cfg.models.dry_run:
        return {
            gripper: _paired_retrieval_average_predict(cfg, query, retrieved, gripper)
            for gripper in (Gripper.GECKO, Gripper.SILICONE)
        }

    raw = get_client(cfg).generate_json(
        system=cfg.prompts.system,
        instruction=cfg.prompts.paired_gripper_instruction,
        schema=PairedGripperPrediction,
        image_bgr=image_bgr,
        extra=payload,
    )
    response = PairedGripperPrediction.model_validate(raw)
    predictions = {
        Gripper.GECKO: response.gecko,
        Gripper.SILICONE: response.silicone,
    }
    for gripper, prediction in predictions.items():
        prediction.candidate_gripper = gripper
        prediction.predicted_normal_force_n = _quantize(
            prediction.predicted_normal_force_n, cfg
        )
    return predictions


def _paired_retrieval_average_predict(
    cfg: Config,
    query: Query,
    retrieved: list[RetrievedObjectExperience],
    gripper: Gripper,
) -> PerGripperPrediction:
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
            reasoning_trace="no feasible paired neighbors",
        )
    weights = np.asarray([max(0.0, item.score) for item in usable], dtype=float)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    forces = np.asarray([getattr(item, force_field) for item in usable], dtype=float)
    estimate = float((weights * forces).sum() / weights.sum())
    return PerGripperPrediction(
        candidate_gripper=gripper,
        feasible=True,
        predicted_normal_force_n=_quantize(estimate, cfg),
        reasoning_trace=(
            "offline similarity-weighted estimate from the shared paired-object neighbors"
        ),
    )


def _stub_prediction(
    cfg: Config,
    query: CandidateQuery,
    retrieved: list[RetrievedExperience],
    physics_estimate: PhysicsEstimate | None,
) -> PerGripperPrediction:
    """Deterministic offline stand-in: prefer physics, then retrieval, then a
    weight-based heuristic. Keeps dry-run runs meaningful, not constant."""
    if physics_estimate is not None:
        feasible = physics_estimate.feasible
        force = (
            physics_estimate.min_force_n
            if (feasible and physics_estimate.min_force_n is not None)
            else cfg.force.limit_n
        )
    elif retrieved:
        rp = retrieval_average_predict(cfg, query, retrieved)
        feasible, force = rp.feasible, rp.predicted_normal_force_n
    else:
        feasible, force = True, _quantize(0.002 * query.mass_g + 0.25, cfg)
    return PerGripperPrediction(
        candidate_gripper=query.candidate_gripper,
        compatibility=Compatibility.UNKNOWN,
        feasible=feasible,
        predicted_normal_force_n=force,
        reasoning_trace="dry-run stub",
    )


# --------------------------------------------------------------------------- #
# Non-VLM estimators
# --------------------------------------------------------------------------- #
def retrieval_average_predict(
    cfg: Config, query: CandidateQuery, retrieved: list[RetrievedExperience]
) -> PerGripperPrediction:
    """E3b: similarity-weighted average of the top-k feasible forces."""
    feasible = [
        r for r in retrieved if r.record.feasible and r.record.min_force_n is not None
    ]
    if not feasible:
        return PerGripperPrediction(
            candidate_gripper=query.candidate_gripper,
            feasible=False,
            predicted_normal_force_n=cfg.force.limit_n,
            reasoning_trace="no feasible neighbours",
        )
    weights = np.array([max(0.0, r.score) for r in feasible], dtype=float)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    forces = np.array([r.record.min_force_n for r in feasible], dtype=float)
    est = float((weights * forces).sum() / weights.sum())
    return PerGripperPrediction(
        candidate_gripper=query.candidate_gripper,
        feasible=True,
        predicted_normal_force_n=_quantize(est, cfg),
        reasoning_trace="similarity-weighted average of retrieved forces",
    )


def physics_predict(
    cfg: Config, query: CandidateQuery, physics_estimate: PhysicsEstimate
) -> PerGripperPrediction:
    """E4: straight from the calibrated physics solver."""
    return PerGripperPrediction(
        candidate_gripper=query.candidate_gripper,
        feasible=physics_estimate.feasible,
        predicted_normal_force_n=(
            physics_estimate.min_force_n
            if (physics_estimate.feasible and physics_estimate.min_force_n is not None)
            else cfg.force.limit_n
        ),
        reasoning_trace="calibrated reduced-order physics",
    )


# --------------------------------------------------------------------------- #
# Deterministic selector
# --------------------------------------------------------------------------- #
def select(
    predictions: dict[Gripper, PerGripperPrediction], reasoning: str = ""
) -> SelectionResult:
    feasible = [p for p in predictions.values() if p.feasible]
    candidate_map = {g.value: p for g, p in predictions.items()}
    if not feasible:
        return SelectionResult(
            desired_gripper="none",
            predicted_normal_force_n=None,
            candidate_predictions=candidate_map,
            reasoning_trace=reasoning or "no feasible gripper",
        )
    minimum = min(predicted.predicted_normal_force_n for predicted in feasible)
    tied = [predicted for predicted in feasible if predicted.predicted_normal_force_n == minimum]
    compatibility_rank = {
        Compatibility.HIGH: 3,
        Compatibility.MEDIUM: 2,
        Compatibility.LOW: 1,
        Compatibility.UNKNOWN: 0,
    }
    best = max(tied, key=lambda predicted: compatibility_rank[predicted.compatibility])
    tie_break_reason = None
    if len(tied) > 1:
        ranks = {compatibility_rank[predicted.compatibility] for predicted in tied}
        tie_break_reason = (
            "higher predicted material compatibility"
            if len(ranks) > 1
            else "stable gripper order because force and compatibility were equal"
        )
    return SelectionResult(
        desired_gripper=best.candidate_gripper.value,
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
    )
