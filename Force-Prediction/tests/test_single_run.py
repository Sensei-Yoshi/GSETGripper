from __future__ import annotations

import pytest

from modules.contracts import ExperienceRecord, Gripper, SelectionResult
from modules.serial_output import SerialSendResult
from streamlit_app.tabs.single_run import (
    _send_selected_force,
    _single_run_training_records,
)


def _record(object_id: str, surface_id: str) -> ExperienceRecord:
    return ExperienceRecord(
        object_id=object_id,
        surface_id=surface_id,
        image_path=f"{object_id}.jpg",
        mass_g=100,
        roughness_index=800,
        projected_contact_fraction=1.0,
        gripper=Gripper.GECKO,
        min_force_n=1.0,
        feasible=True,
        semantic_description="test surface",
    )


def test_single_run_retrieval_uses_only_eligible_non_sibling_references() -> None:
    records = [
        _record("train_other", "train_other"),
        _record("shared", "shared"),
        _record("shared__condition_2", "shared"),
        _record("held_out", "held_out"),
    ]

    training = _single_run_training_records(
        records,
        experiment="e3",
        query_object_id="shared",
        query_surface_id="shared",
        reference_ids=("shared", "shared__condition_2", "train_other"),
    )

    assert [record.object_id for record in training] == ["train_other"]


def test_single_run_non_retrieval_experiments_preserve_query_exclusion() -> None:
    records = [
        _record("query", "query"),
        _record("other", "other"),
    ]

    training = _single_run_training_records(
        records,
        experiment="e1",
        query_object_id="query",
        query_surface_id="query",
        reference_ids=(),
    )

    assert [record.object_id for record in training] == ["other"]


def test_serial_delivery_uses_authoritative_selected_gripper_and_force(monkeypatch) -> None:
    captured = {}

    def fake_send(port, gripper, force_n, limit_n):  # noqa: ANN001, ANN202
        captured.update(
            port=port,
            gripper=gripper,
            force_n=force_n,
            limit_n=limit_n,
        )
        return SerialSendResult(
            port=port,
            gripper=Gripper(gripper),
            force_n=force_n,
        )

    monkeypatch.setattr("streamlit_app.tabs.single_run.send_force", fake_send)
    selection = SelectionResult(
        desired_gripper="silicone",
        predicted_normal_force_n=1.75,
        candidate_predictions={},
    )

    result = _send_selected_force("/dev/fake", selection, 8.0)

    assert result.gripper is Gripper.SILICONE
    assert captured == {
        "port": "/dev/fake",
        "gripper": "silicone",
        "force_n": 1.75,
        "limit_n": 8.0,
    }


def test_serial_delivery_rejects_no_feasible_selection(monkeypatch) -> None:
    monkeypatch.setattr(
        "streamlit_app.tabs.single_run.send_force",
        lambda *_args, **_kwargs: pytest.fail("serial transport must not be called"),
    )
    selection = SelectionResult(
        desired_gripper="none",
        predicted_normal_force_n=None,
        candidate_predictions={},
    )

    with pytest.raises(ValueError, match="did not select"):
        _send_selected_force("/dev/fake", selection, 8.0)
