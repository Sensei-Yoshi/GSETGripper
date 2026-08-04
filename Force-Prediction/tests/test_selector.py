from __future__ import annotations

from modules.config import load_config
from modules.contracts import (
    Compatibility,
    ExperienceRecord,
    Gripper,
    ObjectRecord,
    PerGripperPrediction,
    SelectionResult,
)
from modules.evaluation import EvalRow, compute_metrics
from modules.prediction import clamp_force, select


def _p(gripper, force, feasible=True):
    return PerGripperPrediction(
        candidate_gripper=gripper, feasible=feasible,
        predicted_normal_force_n=force, compatibility=Compatibility.UNKNOWN,
    )


def test_picks_lowest_feasible():
    preds = {Gripper.GECKO: _p(Gripper.GECKO, 1.25), Gripper.SILICONE: _p(Gripper.SILICONE, 2.5)}
    r = select(preds)
    assert r.desired_gripper == "gecko" and r.predicted_normal_force_n == 1.25


def test_force_normalization_preserves_continuous_values_without_upper_cap():
    cfg = load_config()

    assert clamp_force(1.01, cfg) == 1.01
    assert clamp_force(1.234567, cfg) == 1.234567
    assert clamp_force(9.0, cfg) == 9.0
    assert clamp_force(10.506, cfg) == 10.506
    assert clamp_force(-1.0, cfg) == cfg.force.min_n


def test_skips_infeasible_even_if_lower():
    preds = {
        Gripper.GECKO: _p(Gripper.GECKO, 0.5, feasible=False),
        Gripper.SILICONE: _p(Gripper.SILICONE, 3.0),
    }
    r = select(preds)
    assert r.desired_gripper == "silicone"


def test_none_when_all_infeasible():
    preds = {
        Gripper.GECKO: _p(Gripper.GECKO, 8.0, feasible=False),
        Gripper.SILICONE: _p(Gripper.SILICONE, 8.0, feasible=False),
    }
    r = select(preds)
    assert r.desired_gripper == "none" and r.predicted_normal_force_n is None


def test_prediction_tie_is_explicit_and_uses_compatibility():
    gecko = _p(Gripper.GECKO, 1.0)
    silicone = _p(Gripper.SILICONE, 1.0)
    silicone.compatibility = Compatibility.HIGH

    result = select({Gripper.GECKO: gecko, Gripper.SILICONE: silicone})

    assert result.prediction_tie is True
    assert result.desired_gripper == "silicone"
    assert result.tie_break_reason == "higher predicted material compatibility"


def test_model_recommendation_is_diagnostic_not_authoritative():
    predictions = {
        Gripper.GECKO: _p(Gripper.GECKO, 0.8),
        Gripper.SILICONE: _p(Gripper.SILICONE, 1.2),
    }

    result = select(
        predictions,
        model_recommended_gripper="silicone",
        model_recommendation_summary="model preferred silicone",
    )

    assert result.desired_gripper == "gecko"
    assert result.model_recommended_gripper == "silicone"
    assert result.recommendation_agrees_with_selector is False


def test_single_gripper_metrics_exclude_inactive_and_selection_scores():
    cfg = load_config().model_copy(deep=True)
    cfg.prediction.active_grippers = (Gripper.SILICONE,)
    truth = ObjectRecord(
        object_id="single",
        silicone=ExperienceRecord(
            object_id="single",
            image_path="",
            gripper=Gripper.SILICONE,
            min_force_n=2.0,
        ),
    )
    prediction = _p(Gripper.SILICONE, 2.25)
    result = SelectionResult(
        desired_gripper="silicone",
        predicted_normal_force_n=2.25,
        candidate_predictions={"silicone": prediction},
    )

    metrics = compute_metrics(
        [EvalRow(object_id="single", truth=truth, result=result)], cfg
    )

    assert set(metrics.force) == {"silicone", "overall"}
    assert metrics.force["silicone"]["mae"] == 0.25
    assert metrics.selection == {"applicable": False, "n": 0}
    assert metrics.model_recommendation == {"applicable": False, "n": 0}


def test_evaluation_accepts_either_gripper_for_true_force_tie():
    truth = ObjectRecord(
        object_id="tie",
        gecko=ExperienceRecord(object_id="tie", image_path="", mass_g=100,
                               roughness_index=2, projected_contact_fraction=0.8,
                               gripper=Gripper.GECKO, min_force_n=1.0),
        silicone=ExperienceRecord(object_id="tie", image_path="", mass_g=100,
                                  roughness_index=2, projected_contact_fraction=0.8,
                                  gripper=Gripper.SILICONE, min_force_n=1.0),
    )
    predictions = {
        "gecko": _p(Gripper.GECKO, 1.25),
        "silicone": _p(Gripper.SILICONE, 1.0),
    }
    result = SelectionResult(desired_gripper="silicone", predicted_normal_force_n=1.0,
                             candidate_predictions=predictions)
    metrics = compute_metrics([EvalRow(object_id="tie", truth=truth, result=result)], load_config())
    assert metrics.selection["accuracy"] == 1.0
    assert metrics.selection["mean_regret_n"] == 0.0
