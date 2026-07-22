from __future__ import annotations

from force_prediction.config import load_config
from force_prediction.contracts import (
    Compatibility,
    ExperienceRecord,
    Gripper,
    ObjectRecord,
    PerGripperPrediction,
    SelectionResult,
)
from force_prediction.evaluation import EvalRow, compute_metrics
from force_prediction.prediction import select


def _p(gripper, force, feasible=True):
    return PerGripperPrediction(
        candidate_gripper=gripper, feasible=feasible,
        predicted_normal_force_n=force, compatibility=Compatibility.UNKNOWN,
    )


def test_picks_lowest_feasible():
    preds = {Gripper.GECKO: _p(Gripper.GECKO, 1.25), Gripper.SILICONE: _p(Gripper.SILICONE, 2.5)}
    r = select(preds)
    assert r.desired_gripper == "gecko" and r.predicted_normal_force_n == 1.25


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


def test_evaluation_accepts_either_gripper_for_true_force_tie():
    truth = ObjectRecord(
        object_id="tie",
        gecko=ExperienceRecord(object_id="tie", image_path="", mass_g=100,
                               roughness_class=2, projected_contact_fraction=0.8,
                               gripper=Gripper.GECKO, min_force_n=1.0),
        silicone=ExperienceRecord(object_id="tie", image_path="", mass_g=100,
                                  roughness_class=2, projected_contact_fraction=0.8,
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
