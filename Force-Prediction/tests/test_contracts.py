from __future__ import annotations

import pytest
from pydantic import ValidationError

from modules.config import RoughnessConfig, load_config
from modules.contracts import (
    ExperienceRecord,
    Gripper,
    Query,
    group_by_object,
)

CFG = load_config()


def _rec(**kw):
    base = dict(
        object_id="o1", image_path="", mass_g=100.0, roughness_index=2,
        projected_contact_fraction=0.8, gripper=Gripper.GECKO,
        min_force_n=1.0, feasible=True,
    )
    base.update(kw)
    return ExperienceRecord(**base)


def test_feasible_requires_force():
    with pytest.raises(ValidationError):
        _rec(feasible=True, min_force_n=None)


def test_infeasible_requires_limit_and_no_force():
    with pytest.raises(ValidationError):
        _rec(feasible=False, min_force_n=2.0, failed_at_limit_n=8.0)
    ok = _rec(feasible=False, min_force_n=None, failed_at_limit_n=8.0)
    assert ok.failed_at_limit_n == 8.0


@pytest.mark.parametrize("value", [-0.01, float("nan"), float("inf")])
def test_roughness_index_rejects_invalid_values(value):
    with pytest.raises(ValidationError):
        Query(object_id="bad", image_path="", roughness_index=value)


def test_roughness_config_requires_a_positive_scale_and_increasing_direction():
    with pytest.raises(ValidationError):
        RoughnessConfig(
            metric_name="test_index",
            units="unitless_index",
            higher_is_rougher=True,
            characteristic_scale=0,
        )
    with pytest.raises(ValidationError):
        RoughnessConfig(
            metric_name="test_index",
            units="unitless_index",
            higher_is_rougher=False,
            characteristic_scale=250,
        )


def test_group_and_oracle():
    recs = [
        _rec(object_id="o1", gripper=Gripper.GECKO, min_force_n=1.25),
        _rec(object_id="o1", gripper=Gripper.SILICONE, min_force_n=2.5),
    ]
    objects = group_by_object(recs)
    obj = objects["o1"]
    g, f = obj.oracle()
    assert g is Gripper.GECKO and f == 1.25
