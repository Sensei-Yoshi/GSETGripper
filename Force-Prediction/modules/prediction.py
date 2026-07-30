"""Shared VLM force estimation and the authoritative deterministic selector.

E1 through E6 use one joint VLM response for both grippers. Python always makes the final
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
from .models.gemini import get_client
from .retrieval import (
    MeasurementField,
    RankingFeature,
    RetrievalMode,
    RetrievedObjectExperience,
    default_ranking_features,
    normalized_weights,
)
from .roughness_representation import binary_roughness_category

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
        payload["mass_g"] = query.mass_g
        if cfg.inputs.use_roughness:
            if query.roughness_index is None:
                raise ValueError("enabled measured roughness requires a query value")
            if cfg.inputs.roughness_representation == "binary":
                payload["roughness_category"] = binary_roughness_category(
                    query.roughness_index,
                    cfg.roughness.binary_threshold,
                )
            else:
                payload["roughness_index"] = query.roughness_index
        if cfg.inputs.use_projected_contact:
            payload["projected_contact_fraction"] = query.projected_contact_fraction
    return payload


def embodiment_payload(
    cfg: Config,
    active_grippers: tuple[Gripper, ...] = GRIPPERS,
) -> dict[str, str]:
    return {
        gripper.value: cfg.embodiments[gripper.value].description
        for gripper in active_grippers
    }


def _generation_payload(
    cfg: Config,
    query: Query,
    retrieved: list[RetrievedObjectExperience],
    *,
    active_grippers: tuple[Gripper, ...],
    include_measured: bool,
    include_retrieval: bool,
    retrieval_mode: RetrievalMode | None,
    ranking_features: tuple[RankingFeature, ...] = (),
    visible_condition_fields: tuple[MeasurementField, ...] = (),
) -> dict:
    payload: dict = {
        "query": _query_payload(query, cfg, include_measured=include_measured),
        "active_grippers": [gripper.value for gripper in active_grippers],
        "gripper_embodiments": embodiment_payload(cfg, active_grippers),
        "force_constraints": _force_constraints(cfg),
    }
    if include_measured and cfg.inputs.use_roughness:
        if cfg.inputs.roughness_representation == "binary":
            payload["roughness_measurement"] = {
                "representation": "binary_category",
                "categories": ["smooth", "rough"],
                "scope": "relative_to_this_dataset",
                "numeric_values_withheld": True,
            }
        else:
            payload["roughness_measurement"] = {
                "metric_name": cfg.roughness.metric_name,
                "units": cfg.roughness.units,
                "higher_is_rougher": cfg.roughness.higher_is_rougher,
                "characteristic_scale": cfg.roughness.characteristic_scale,
            }
    if not include_retrieval:
        return payload

    mode = retrieval_mode or RetrievalMode.HYBRID
    payload["query_semantic_description"] = query.semantic_description
    payload["retrieved_objects"] = _group_retrieved_surfaces(
        cfg,
        retrieved,
        mode=mode,
        active_grippers=active_grippers,
    )
    if mode is RetrievalMode.SEMANTIC_ONLY:
        payload["retrieval_config"] = {
            "mode": mode.value,
            "k": cfg.retrieval.k,
            "score": "cosine_semantic_embedding_only",
        }
    elif cfg.inputs.roughness_representation == "binary" and cfg.inputs.use_roughness:
        payload["retrieval_config"] = {
            "mode": mode.value,
            "k": cfg.retrieval.k,
            "vlm_roughness_representation": "binary_category",
            "ranking_note": (
                "Neighbors were ranked upstream with the configured hybrid retrieval. "
                "Continuous roughness values are intentionally withheld; do not infer them."
            ),
        }
    else:
        active_ranking_features = (
            ranking_features or default_ranking_features(cfg)
        )
        payload["retrieval_config"] = {
            "mode": mode.value,
            "k": cfg.retrieval.k,
            "ranking_features": list(active_ranking_features),
            "visible_condition_fields": list(visible_condition_fields),
            "condition_policy": "baseline_plus_visible_controlled_variants",
            "conditions_per_surface": cfg.retrieval.conditions_per_surface,
            "projected_contact_used_for_surface_ranking": (
                "contact" in active_ranking_features
            ),
            "normalized_weights": normalized_weights(
                cfg, active_ranking_features
            ),
            "sigma_mass": cfg.retrieval.sigma_mass,
            "roughness_characteristic_scale": cfg.roughness.characteristic_scale,
        }
        if "contact" in active_ranking_features:
            payload["retrieval_config"]["sigma_contact"] = (
                cfg.retrieval.sigma_contact
            )
    return payload


def _group_retrieved_surfaces(
    cfg: Config,
    retrieved: list[RetrievedObjectExperience],
    *,
    mode: RetrievalMode,
    active_grippers: tuple[Gripper, ...],
) -> list[dict]:
    """Create the version-3 VLM payload: one surface plus controlled conditions."""
    grouped: dict[str, list[RetrievedObjectExperience]] = {}
    for item in retrieved:
        grouped.setdefault(item.surface_id or item.object_id, []).append(item)
    output: list[dict] = []
    for surface_id, conditions in sorted(
        grouped.items(), key=lambda pair: pair[1][0].surface_rank
    ):
        conditions.sort(key=lambda item: item.condition_rank)
        best = max(conditions, key=lambda item: item.score)
        surface_score = max(
            item.surface_score if item.surface_score is not None else item.score
            for item in conditions
        )
        condition_payloads: list[dict] = []
        for item in conditions:
            raw = item.to_payload(
                mode=mode,
                active_grippers=active_grippers,
                include_roughness=cfg.inputs.use_roughness,
                include_contact=cfg.inputs.use_projected_contact,
            )
            if mode is RetrievalMode.HYBRID:
                raw.pop("semantic_description", None)
                raw.pop("surface_id", None)
                raw.pop("rank", None)
                raw.pop("surface_rank", None)
                raw.pop("surface_score", None)
                raw.pop("image_path", None)
                # Projected contact is a within-surface control, not a cross-object
                # feature: a neighbor baseline's absolute contact fraction is the
                # confounded cross-object signal E6 deliberately excludes. Expose it
                # only on the query and as within-surface variant deltas
                # (comparison_to_baseline + force_deltas), never as a neighbor
                # baseline's absolute value the model can spuriously match against.
                if item.condition_role == "baseline":
                    raw.pop("projected_contact_fraction", None)
                if (
                    cfg.inputs.use_roughness
                    and cfg.inputs.roughness_representation == "binary"
                ):
                    if item.roughness_index is None:
                        raise ValueError(
                            "binary VLM roughness requires a retrieved roughness index"
                        )
                    raw.pop("roughness_index", None)
                    raw.pop("score", None)
                    raw["roughness_category"] = binary_roughness_category(
                        item.roughness_index,
                        cfg.roughness.binary_threshold,
                    )
                    similarity = raw.get("similarity")
                    if isinstance(similarity, dict):
                        similarity.pop("roughness", None)
                        similarity.pop("roughness_contribution", None)
                        similarity.pop("total", None)
            condition_payloads.append(raw)
        surface = {
            "schema_version": 3,
            "rank": best.surface_rank,
            "surface_id": surface_id,
            "semantic_description": best.semantic_description,
            "semantic_similarity": best.similarity.semantic,
            "conditions": condition_payloads,
        }
        if not (
            cfg.inputs.use_roughness
            and cfg.inputs.roughness_representation == "binary"
        ):
            surface["score"] = surface_score
        # Compatibility summary for existing result viewers. The nested conditions
        # remain authoritative and contain all retained sibling observations.
        best_payload = condition_payloads[0]
        for key, value in best_payload.items():
            if key not in {
                "object_id",
                "condition_id",
                "condition_rank",
                "condition_role",
                "comparison_to_baseline",
            }:
                surface.setdefault(key, value)
        output.append(surface)
    return output


def vlm_predict_joint(
    cfg: Config,
    query: Query,
    image_bgr: np.ndarray | None,
    retrieved: list[RetrievedObjectExperience],
    *,
    instruction: str,
    include_measured: bool,
    include_retrieval: bool,
    retrieval_mode: RetrievalMode | None = None,
    ranking_features: tuple[RankingFeature, ...] = (),
    visible_condition_fields: tuple[MeasurementField, ...] = (),
) -> JointGripperPrediction:
    """Estimate both grippers and recommend one with exactly one force-generation call."""
    if image_bgr is None:
        raise ValueError("Gemini force prediction requires a decodable object image")
    payload = _generation_payload(
        cfg,
        query,
        retrieved,
        active_grippers=GRIPPERS,
        include_measured=include_measured,
        include_retrieval=include_retrieval,
        retrieval_mode=retrieval_mode,
        ranking_features=ranking_features,
        visible_condition_fields=visible_condition_fields,
    )

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


def vlm_predict_single(
    cfg: Config,
    query: Query,
    image_bgr: np.ndarray | None,
    retrieved: list[RetrievedObjectExperience],
    *,
    gripper: Gripper,
    instruction: str,
    include_measured: bool,
    include_retrieval: bool,
    retrieval_mode: RetrievalMode | None = None,
    ranking_features: tuple[RankingFeature, ...] = (),
    visible_condition_fields: tuple[MeasurementField, ...] = (),
) -> PerGripperPrediction:
    """Estimate one requested gripper with the per-gripper response schema."""
    if image_bgr is None:
        raise ValueError("Gemini force prediction requires a decodable object image")
    payload = _generation_payload(
        cfg,
        query,
        retrieved,
        active_grippers=(gripper,),
        include_measured=include_measured,
        include_retrieval=include_retrieval,
        retrieval_mode=retrieval_mode,
        ranking_features=ranking_features,
        visible_condition_fields=visible_condition_fields,
    )
    raw = get_client(cfg).generate_json(
        system=cfg.prompts.prediction_system,
        instruction=instruction,
        schema=PerGripperPrediction,
        image_bgr=image_bgr,
        extra=payload,
    )
    response = PerGripperPrediction.model_validate(raw)
    response.candidate_gripper = gripper
    response.predicted_normal_force_n = clamp_force(
        response.predicted_normal_force_n, cfg
    )
    return response


def predictions_from_joint(
    response: JointGripperPrediction,
) -> dict[Gripper, PerGripperPrediction]:
    return {
        Gripper.GECKO: response.gecko,
        Gripper.SILICONE: response.silicone,
    }


def select(
    predictions: dict[Gripper, PerGripperPrediction],
    reasoning: str = "",
    *,
    model_recommended_gripper: GripperChoice | None = None,
    model_comparison_evidence: list[str] | None = None,
    model_recommendation_summary: str | None = None,
) -> SelectionResult:
    """Choose the lowest-force feasible gripper; VLM recommendations are diagnostic only."""
    feasible = [prediction for prediction in predictions.values() if prediction.feasible]
    candidate_map = {gripper.value: prediction for gripper, prediction in predictions.items()}

    if not feasible:
        desired_gripper: GripperChoice = "none"
        return SelectionResult(
            desired_gripper=desired_gripper,
            predicted_normal_force_n=None,
            candidate_predictions=candidate_map,
            reasoning_trace=reasoning or "no feasible gripper",
            model_recommended_gripper=model_recommended_gripper,
            model_comparison_evidence=model_comparison_evidence or [],
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
    desired_gripper = (
        "gecko" if best.candidate_gripper is Gripper.GECKO else "silicone"
    )
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
        model_comparison_evidence=model_comparison_evidence or [],
        model_recommendation_summary=model_recommendation_summary,
        recommendation_agrees_with_selector=(
            desired_gripper == model_recommended_gripper
            if model_recommended_gripper is not None
            else None
        ),
    )
