from __future__ import annotations

import pytest

from force_prediction.config import load_config
from force_prediction.contracts import (
    Gripper,
    PairedGripperPrediction,
    PerGripperPrediction,
    Query,
    group_by_object,
)
from force_prediction.hardware import fabricate_records
from force_prediction.pipeline import Pipeline, query_input_from_object
from force_prediction.retrieval import ExperienceIndex, build_embedding_text


def test_paired_delta_matches_truth():
    cfg = load_config()
    cfg.models.dry_run = True
    records = fabricate_records(cfg, 20)
    objects = group_by_object(records)
    index = ExperienceIndex(cfg).fit(records)
    q = Query(object_id="probe", image_path="", mass_g=300, roughness_class=2,
              projected_contact_fraction=0.8, semantic_description="x")
    qv = index.provider.embed(build_embedding_text("x", 300, 2, 0.8, cfg))
    for r in index.retrieve(q, qv, Gripper.GECKO):
        truth_other = objects[r.record.object_id].other_gripper_force(Gripper.GECKO)
        assert r.other_gripper_min_force_n == truth_other


def test_e5_pipeline_runs_end_to_end():
    cfg = load_config()
    cfg.models.dry_run = True
    records = fabricate_records(cfg, 30)
    held = records[0].object_id
    train = [r for r in records if r.object_id != held]
    test = [r for r in records if r.object_id == held]
    pipe = Pipeline(cfg, cfg.experiment("e5")).fit(train)
    result = pipe.predict(query_input_from_object(test, cfg))
    assert result.desired_gripper in ("gecko", "silicone", "none")
    assert set(result.candidate_predictions) == {"gecko", "silicone"}


@pytest.mark.parametrize("experiment", ["e1", "e2", "e3", "e3b", "e4", "e5", "e6"])
def test_every_experiment_runs_offline_with_continuous_forces(experiment):
    cfg = load_config().model_copy(deep=True)
    cfg.models.dry_run = True
    records = fabricate_records(cfg, 24)
    held = records[0].object_id
    train = [record for record in records if record.object_id != held]
    test = [record for record in records if record.object_id == held]

    result = Pipeline(cfg, cfg.experiment(experiment)).fit(train).predict(
        query_input_from_object(test, cfg)
    )

    assert set(result.candidate_predictions) == {"gecko", "silicone"}
    assert all(
        cfg.force.min_n <= prediction.predicted_normal_force_n <= cfg.force.limit_n
        for prediction in result.candidate_predictions.values()
    )


def test_detailed_pipeline_preserves_selection_and_exposes_trace():
    cfg = load_config().model_copy(deep=True)
    cfg.models.dry_run = True
    cfg.retrieval.k = 5
    records = fabricate_records(cfg, 30)
    held = records[0].object_id
    train = [r for r in records if r.object_id != held]
    test = [r for r in records if r.object_id == held]
    pipe = Pipeline(cfg, cfg.experiment("e5")).fit(train)
    query = query_input_from_object(test, cfg)

    detailed = pipe.predict_detailed(query)
    ordinary = pipe.predict(query)

    assert detailed.selection == ordinary
    assert set(detailed.retrieved) == {"gecko", "silicone"}
    assert all(not items for items in detailed.retrieved.values())
    assert len(detailed.retrieved_objects) == 5
    assert len({item.object_id for item in detailed.retrieved_objects}) == 5
    assert all(item.gecko_min_force_n is not None for item in detailed.retrieved_objects)
    assert all(item.silicone_min_force_n is not None for item in detailed.retrieved_objects)
    assert set(detailed.physics_estimates) == {"gecko", "silicone"}


def test_e5_uses_one_object_retrieval_and_one_joint_vlm_call(monkeypatch):
    cfg = load_config().model_copy(deep=True)
    cfg.models.dry_run = False
    cfg.retrieval.embedding.provider = "mock"
    cfg.retrieval.k = 5
    records = fabricate_records(cfg, 20)
    held = records[0].object_id
    train = [record for record in records if record.object_id != held]
    test = [record for record in records if record.object_id == held]
    captured = {}

    class CountingClient:
        generation_calls = 0

        def generate_json(self, **kwargs):
            self.generation_calls += 1
            captured.update(kwargs)
            return PairedGripperPrediction(
                gecko=PerGripperPrediction(
                    candidate_gripper=Gripper.SILICONE,
                    predicted_normal_force_n=0.8,
                    reasoning_trace="joint gecko evidence",
                ),
                silicone=PerGripperPrediction(
                    candidate_gripper=Gripper.GECKO,
                    predicted_normal_force_n=1.3,
                    reasoning_trace="joint silicone evidence",
                ),
            ).model_dump(mode="json")

        def cache_stats(self):
            return {"backend_attempts": {"generation": self.generation_calls, "embedding": 0}}

    client = CountingClient()
    monkeypatch.setattr("force_prediction.prediction.get_client", lambda _cfg: client)
    monkeypatch.setattr("force_prediction.pipeline.get_client", lambda _cfg: client)

    pipe = Pipeline(cfg, cfg.experiment("e5")).fit(train)
    detailed = pipe.predict_detailed(query_input_from_object(test, cfg))

    assert client.generation_calls == 1
    assert captured["schema"] is PairedGripperPrediction
    assert captured["instruction"] == cfg.prompts.experiments["e5"]
    paired_payload = captured["extra"]["retrieved_objects"]
    assert len(paired_payload) == 5
    assert all("gecko_min_force_n" in item for item in paired_payload)
    assert all("silicone_min_force_n" in item for item in paired_payload)
    assert all("image_path" not in item for item in paired_payload)
    assert "retrieved_experiences" not in captured["extra"]
    assert detailed.selection.candidate_predictions["gecko"].candidate_gripper is Gripper.GECKO
    assert detailed.selection.candidate_predictions["silicone"].candidate_gripper is Gripper.SILICONE
    assert detailed.selection.candidate_predictions["gecko"].predicted_normal_force_n == 0.8
    assert detailed.selection.candidate_predictions["silicone"].predicted_normal_force_n == 1.3


def test_e5_contact_ablation_removes_contact_from_vlm_payload(monkeypatch):
    cfg = load_config().model_copy(deep=True)
    cfg.models.dry_run = False
    cfg.retrieval.embedding.provider = "mock"
    cfg.retrieval.use_projected_contact = False
    records = fabricate_records(cfg, 12)
    held = records[0].object_id
    captured = {}

    class CapturingClient:
        def generate_json(self, **kwargs):
            captured.update(kwargs)
            return PairedGripperPrediction(
                gecko=PerGripperPrediction(
                    candidate_gripper=Gripper.GECKO,
                    predicted_normal_force_n=1.0,
                ),
                silicone=PerGripperPrediction(
                    candidate_gripper=Gripper.SILICONE,
                    predicted_normal_force_n=1.0,
                ),
            ).model_dump(mode="json")

        def cache_stats(self):
            return {}

    client = CapturingClient()
    monkeypatch.setattr("force_prediction.prediction.get_client", lambda _cfg: client)
    monkeypatch.setattr("force_prediction.pipeline.get_client", lambda _cfg: client)

    train = [record for record in records if record.object_id != held]
    test = [record for record in records if record.object_id == held]
    Pipeline(cfg, cfg.experiment("e5")).fit(train).predict_detailed(
        query_input_from_object(test, cfg)
    )

    assert "projected_contact_fraction" not in captured["extra"]["query"]
    assert all(
        "projected_contact_fraction" not in item
        for item in captured["extra"]["retrieved_objects"]
    )
