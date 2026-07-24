from __future__ import annotations

from force_prediction.config import load_config
from force_prediction.contracts import Gripper
from force_prediction.learning import base_features
from force_prediction.physics import PhysicsModel, PhysicsParams, weight_n

CFG = load_config()
MODEL = PhysicsModel(PhysicsParams.from_config(CFG), CFG)


def test_silicone_closed_form_matches_solver():
    # T_sil = alpha*a*N ; at the reported min force, holding ~ weight.
    est = MODEL.min_force(Gripper.SILICONE, 300, 2, 0.8)
    assert est.feasible and est.min_force_n is not None
    held = MODEL.holding_force(Gripper.SILICONE, est.raw_force_n, 2, 0.8)
    assert abs(held - weight_n(300, CFG.force.gravity)) < 1e-6


def test_gecko_holding_increases_with_force():
    vals = [MODEL.holding_force(Gripper.GECKO, n, 2, 0.8) for n in (0.5, 1.0, 2.0, 4.0)]
    assert vals == sorted(vals)


def test_min_force_increases_with_mass():
    light = MODEL.min_force(Gripper.SILICONE, 100, 2, 0.8)
    heavy = MODEL.min_force(Gripper.SILICONE, 800, 2, 0.8)
    assert light.feasible and heavy.feasible
    assert heavy.min_force_n >= light.min_force_n


def test_infeasible_when_over_limit():
    est = MODEL.min_force(Gripper.SILICONE, 5000, 5, 0.3)
    assert est.feasible is False and est.min_force_n is None


def test_physics_returns_continuous_solution():
    est = MODEL.min_force(Gripper.GECKO, 250, 1, 0.9)
    if est.feasible:
        assert est.min_force_n == est.raw_force_n


def test_residual_contact_feature_is_an_explicit_ablation():
    with_contact = base_features(250, 2, 0.7, 1.3, include_contact=True)
    without_contact = base_features(250, 2, 0.7, 1.3, include_contact=False)

    assert with_contact == [*without_contact[:2], 0.7, without_contact[-1]]
