"""Per-gripper force estimators + the deterministic selector.

Three estimators, all returning the same `PerGripperPrediction` contract so the
pipeline can swap them by experiment toggle without branching logic elsewhere:

  vlm_predict_gripper       — Gemini structured output (E1/E2/E3/E5); a
                              physics/retrieval-grounded stub under dry_run.
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
    PerGripperPrediction,
    SelectionResult,
)
from .llm import get_client
from .physics import PhysicsEstimate
from .retrieval import RetrievedExperience


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
        "physics_force_estimate_n": (
            physics_estimate.min_force_n if (physics_estimate and include_measured) else None
        ),
        "physics_feasible": (
            physics_estimate.feasible if (physics_estimate and include_measured) else None
        ),
        "retrieved_experiences": [r.to_payload(include_paired) for r in retrieved],
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
    best = min(feasible, key=lambda p: p.predicted_normal_force_n)
    return SelectionResult(
        desired_gripper=best.candidate_gripper.value,
        predicted_normal_force_n=best.predicted_normal_force_n,
        candidate_predictions=candidate_map,
        reasoning_trace=reasoning or "lowest feasible predicted stationary-finger force",
    )
