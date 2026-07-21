from __future__ import annotations

from force_prediction.contracts import Compatibility, Gripper, PerGripperPrediction
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
