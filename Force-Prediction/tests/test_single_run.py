from __future__ import annotations

from modules.contracts import ExperienceRecord, Gripper
from streamlit_app.tabs.single_run import _single_run_training_records


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
        experiment="e4",
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
