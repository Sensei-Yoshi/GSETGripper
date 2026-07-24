from __future__ import annotations

import numpy as np
import pytest

from force_prediction.config import EXPERIMENT_IDS, load_config
from force_prediction.contracts import (
    Gripper,
    JointGripperPrediction,
    PerGripperPrediction,
)
from force_prediction.hardware import fabricate_records
from force_prediction.pipeline import Pipeline, query_input_from_object
from force_prediction.prediction import clamp_force
from force_prediction.retrieval import ExperienceIndex


@pytest.mark.parametrize("experiment", EXPERIMENT_IDS)
def test_every_experiment_runs_offline_with_continuous_forces(experiment):
    cfg = load_config().model_copy(deep=True)
    cfg.models.dry_run = True
    records = fabricate_records(cfg, 24)
    held = records[0].object_id
    train = [record for record in records if record.object_id != held]
    test = [record for record in records if record.object_id == held]

    result = Pipeline(cfg, experiment).fit(train).predict(query_input_from_object(test, cfg))

    assert set(result.candidate_predictions) == {"gecko", "silicone"}
    assert all(
        cfg.force.min_n <= prediction.predicted_normal_force_n <= cfg.force.limit_n
        for prediction in result.candidate_predictions.values()
    )


def test_e4_detailed_result_contains_one_shared_top_k_list():
    cfg = load_config().model_copy(deep=True)
    cfg.models.dry_run = True
    records = fabricate_records(cfg, 30)
    held = records[0].object_id
    train = [record for record in records if record.object_id != held]
    test = [record for record in records if record.object_id == held]

    detailed = Pipeline(cfg, "e4").fit(train).predict_detailed(
        query_input_from_object(test, cfg)
    )

    assert detailed.experiment_id == "e4"
    assert detailed.experiment_method == "paired_retrieval_vlm"
    assert len(detailed.retrieved_objects) == cfg.retrieval.k
    assert len({item.object_id for item in detailed.retrieved_objects}) == cfg.retrieval.k
    assert all(item.object_id != held for item in detailed.retrieved_objects)
    assert all(item.gecko_min_force_n is not None for item in detailed.retrieved_objects)
    assert all(item.silicone_min_force_n is not None for item in detailed.retrieved_objects)
    assert detailed.physics_estimates == {}


def test_e4_uses_one_object_retrieval_and_one_joint_vlm_call(monkeypatch):
    cfg = load_config().model_copy(deep=True)
    cfg.models.dry_run = False
    cfg.retrieval.embedding.provider = "mock"
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
            return JointGripperPrediction(
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
                recommended_gripper="silicone",
                recommendation_summary="model preferred silicone",
            ).model_dump(mode="json")

        def cache_stats(self):
            return {"backend_attempts": {"generation": self.generation_calls, "embedding": 0}}

    client = CountingClient()
    retrieval_calls = 0
    original_retrieve = ExperienceIndex.retrieve_objects

    def counted_retrieve(self, *args, **kwargs):
        nonlocal retrieval_calls
        retrieval_calls += 1
        return original_retrieve(self, *args, **kwargs)

    monkeypatch.setattr("force_prediction.prediction.get_client", lambda _cfg: client)
    monkeypatch.setattr("force_prediction.experiments.get_client", lambda _cfg: client)
    monkeypatch.setattr(ExperienceIndex, "retrieve_objects", counted_retrieve)

    detailed = Pipeline(cfg, "e4").fit(train).predict_detailed(
        query_input_from_object(test, cfg)
    )

    assert client.generation_calls == 1
    assert retrieval_calls == 1
    assert captured["schema"] is JointGripperPrediction
    assert captured["instruction"] == cfg.prompts.experiments["e4"]
    paired_payload = captured["extra"]["retrieved_objects"]
    assert len(paired_payload) == cfg.retrieval.k
    assert all("gecko_min_force_n" in item for item in paired_payload)
    assert all("silicone_min_force_n" in item for item in paired_payload)
    assert all("image_path" not in item for item in paired_payload)
    assert "retrieved_experiences" not in captured["extra"]
    assert detailed.selection.desired_gripper == "gecko"
    assert detailed.selection.model_recommended_gripper == "silicone"
    assert detailed.selection.recommendation_agrees_with_selector is False


def test_e4_contact_ablation_removes_contact_from_joint_payload(monkeypatch):
    cfg = load_config().model_copy(deep=True)
    cfg.models.dry_run = False
    cfg.retrieval.embedding.provider = "mock"
    cfg.inputs.use_projected_contact = False
    records = fabricate_records(cfg, 12)
    held = records[0].object_id
    captured = {}

    class CapturingClient:
        def generate_json(self, **kwargs):
            captured.update(kwargs)
            return JointGripperPrediction(
                gecko=PerGripperPrediction(
                    candidate_gripper=Gripper.GECKO,
                    predicted_normal_force_n=1.0,
                ),
                silicone=PerGripperPrediction(
                    candidate_gripper=Gripper.SILICONE,
                    predicted_normal_force_n=1.0,
                ),
                recommended_gripper="gecko",
            ).model_dump(mode="json")

        def cache_stats(self):
            return {}

    client = CapturingClient()
    monkeypatch.setattr("force_prediction.prediction.get_client", lambda _cfg: client)
    monkeypatch.setattr("force_prediction.experiments.get_client", lambda _cfg: client)
    train = [record for record in records if record.object_id != held]
    test = [record for record in records if record.object_id == held]

    Pipeline(cfg, "e4").fit(train).predict_detailed(query_input_from_object(test, cfg))

    assert "projected_contact_fraction" not in captured["extra"]["query"]
    assert all(
        "projected_contact_fraction" not in item
        for item in captured["extra"]["retrieved_objects"]
    )
    assert captured["extra"]["retrieval_config"]["k"] == cfg.retrieval.k


def test_e5_loads_only_calibrated_physics(monkeypatch):
    cfg = load_config().model_copy(deep=True)
    records = fabricate_records(cfg, 20)
    held = records[0].object_id
    train = [record for record in records if record.object_id != held]
    test = [record for record in records if record.object_id == held]

    def unexpected(*_args, **_kwargs):
        raise AssertionError("E5 must not initialize VLM, retrieval, or embedding resources")

    monkeypatch.setattr("force_prediction.experiments.get_embedding_provider", unexpected)
    monkeypatch.setattr("force_prediction.experiments.vlm_predict_joint", unexpected)
    monkeypatch.setattr("force_prediction.experiments.describe", unexpected)
    detailed = Pipeline(cfg, "e5").fit(train).predict_detailed(
        query_input_from_object(test, cfg)
    )

    assert detailed.experiment_method == "calibrated_physics"
    assert detailed.retrieved_objects == []
    assert set(detailed.physics_estimates) == {"gecko", "silicone"}
    assert detailed.selection.model_recommended_gripper is None


def test_e6_is_the_same_e5_physics_plus_the_learned_residual(monkeypatch):
    cfg = load_config().model_copy(deep=True)
    cfg.learning.embedding_pca_dims = 0
    records = fabricate_records(cfg, 24)
    held = records[0].object_id
    train = [record for record in records if record.object_id != held]
    test = [record for record in records if record.object_id == held]
    query = query_input_from_object(test, cfg)

    class ConstantResidual:
        def __init__(self, _cfg):
            pass

        def fit(self, base, embeddings, residuals):
            assert len(base) == len(residuals)
            assert embeddings.shape[1] == 0
            return self

        def predict_residual(self, base, embeddings):
            assert len(base) == 1
            assert embeddings.shape == (1, 0)
            return np.asarray([0.2])

    monkeypatch.setattr("force_prediction.experiments.ResidualForceModel", ConstantResidual)
    e5 = Pipeline(cfg, "e5").fit(train).predict_detailed(query)
    e6 = Pipeline(cfg, "e6").fit(train).predict_detailed(query)

    assert e6.physics_estimates == e5.physics_estimates
    for gripper in ("gecko", "silicone"):
        e5_prediction = e5.selection.candidate_predictions[gripper]
        e6_prediction = e6.selection.candidate_predictions[gripper]
        if e5_prediction.feasible:
            assert e6_prediction.predicted_normal_force_n == pytest.approx(
                clamp_force(e5_prediction.predicted_normal_force_n + 0.2, cfg)
            )


def test_e6_keeps_semantic_embeddings_without_vlm_force_or_retrieval(monkeypatch):
    cfg = load_config().model_copy(deep=True)
    cfg.models.dry_run = True
    records = fabricate_records(cfg, 24)
    held = records[0].object_id
    train = [record for record in records if record.object_id != held]
    test = [record for record in records if record.object_id == held]

    def unexpected(*_args, **_kwargs):
        raise AssertionError("E6 must not initialize retrieval or call the force VLM")

    monkeypatch.setattr("force_prediction.experiments.ExperienceIndex", unexpected)
    monkeypatch.setattr("force_prediction.experiments.vlm_predict_joint", unexpected)
    pipeline = Pipeline(cfg, "e6").fit(train)
    detailed = pipeline.predict_detailed(query_input_from_object(test, cfg))

    assert pipeline.strategy.provider is not None
    assert detailed.retrieved_objects == []
    assert detailed.selection.model_recommended_gripper is None
