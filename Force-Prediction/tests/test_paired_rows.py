from __future__ import annotations

import numpy as np
import pytest

from modules.config import EXPERIMENT_IDS, load_config
from modules.contracts import (
    Gripper,
    JointGripperPrediction,
    PerGripperPrediction,
)
from modules.hardware import fabricate_records
from modules.pipeline import Pipeline, query_input_from_object
from modules.retrieval import ExperienceIndex, RetrievalMode
from tests.fakes import FakeEmbeddingProvider, install_gemini_fakes


def _query_with_image(records, cfg):
    query = query_input_from_object(records, cfg)
    query.image_bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    return query


@pytest.mark.parametrize("experiment", EXPERIMENT_IDS)
def test_every_experiment_runs_with_explicit_gemini_fake_and_continuous_forces(
    experiment, monkeypatch
):
    cfg = load_config().model_copy(deep=True)
    install_gemini_fakes(monkeypatch, cfg.retrieval.embedding.dim)
    records = fabricate_records(cfg, 24)
    held = records[0].object_id
    train = [record for record in records if record.object_id != held]
    test = [record for record in records if record.object_id == held]

    result = Pipeline(cfg, experiment).fit(train).predict(_query_with_image(test, cfg))

    assert set(result.candidate_predictions) == {"gecko", "silicone"}
    assert all(
        cfg.force.min_n <= prediction.predicted_normal_force_n <= cfg.force.limit_n
        for prediction in result.candidate_predictions.values()
    )


@pytest.mark.parametrize("experiment", EXPERIMENT_IDS)
@pytest.mark.parametrize("gripper", (Gripper.GECKO, Gripper.SILICONE))
def test_every_experiment_supports_one_active_gripper(
    experiment, gripper, monkeypatch
):
    cfg = load_config().model_copy(deep=True)
    cfg.prediction.active_grippers = (gripper,)
    client = install_gemini_fakes(monkeypatch, cfg.retrieval.embedding.dim)
    records = fabricate_records(cfg, 16)
    held = records[0].object_id
    train = [record for record in records if record.object_id != held]
    test = [record for record in records if record.object_id == held]

    detailed = Pipeline(cfg, experiment).fit(train).predict_detailed(
        _query_with_image(test, cfg)
    )

    assert set(detailed.selection.candidate_predictions) == {gripper.value}
    assert detailed.active_grippers == (gripper.value,)
    assert detailed.generation_mode == "single"
    assert detailed.selection.model_recommended_gripper is None
    assert client.generation_calls == 1


def test_single_silicone_e4_uses_per_gripper_schema_and_filtered_payload(monkeypatch):
    cfg = load_config().model_copy(deep=True)
    cfg.prediction.active_grippers = (Gripper.SILICONE,)
    records = fabricate_records(cfg, 16)
    held = records[0].object_id
    captured = {}

    class CapturingClient:
        def generate_json(self, **kwargs):
            captured.update(kwargs)
            return PerGripperPrediction(
                candidate_gripper=Gripper.GECKO,
                predicted_normal_force_n=1.125,
            ).model_dump(mode="json")

        def cache_stats(self):
            return {}

    client = CapturingClient()
    monkeypatch.setattr("modules.prediction.get_client", lambda _cfg: client)
    monkeypatch.setattr("modules.experiments.helper.get_client", lambda _cfg: client)
    monkeypatch.setattr(
        "modules.experiments.helper.get_embedding_provider",
        lambda _cfg: FakeEmbeddingProvider(cfg.retrieval.embedding.dim),
    )
    train = [record for record in records if record.object_id != held]
    test = [record for record in records if record.object_id == held]

    detailed = Pipeline(cfg, "e4").fit(train).predict_detailed(
        _query_with_image(test, cfg)
    )

    assert captured["schema"] is PerGripperPrediction
    assert captured["extra"]["active_grippers"] == ["silicone"]
    assert set(captured["extra"]["gripper_embodiments"]) == {"silicone"}
    assert all(
        "silicone_min_force_n" in item and "gecko_min_force_n" not in item
        for item in captured["extra"]["retrieved_objects"]
    )
    prediction = detailed.selection.candidate_predictions["silicone"]
    assert prediction.candidate_gripper is Gripper.SILICONE
    assert prediction.predicted_normal_force_n == 1.125


def test_e4_detailed_result_contains_one_shared_top_k_list(monkeypatch):
    cfg = load_config().model_copy(deep=True)
    install_gemini_fakes(monkeypatch, cfg.retrieval.embedding.dim)
    records = fabricate_records(cfg, 30)
    held = records[0].object_id
    train = [record for record in records if record.object_id != held]
    test = [record for record in records if record.object_id == held]

    detailed = Pipeline(cfg, "e4").fit(train).predict_detailed(
        _query_with_image(test, cfg)
    )

    assert detailed.experiment_id == "e4"
    assert detailed.experiment_method == "hybrid_retrieval_vlm"
    assert len(detailed.retrieved_objects) == cfg.retrieval.k
    assert len({item.object_id for item in detailed.retrieved_objects}) == cfg.retrieval.k
    assert all(item.object_id != held for item in detailed.retrieved_objects)
    assert all(item.gecko_min_force_n is not None for item in detailed.retrieved_objects)
    assert all(item.silicone_min_force_n is not None for item in detailed.retrieved_objects)
    assert detailed.physics_estimates == {}


def test_e4_uses_one_object_retrieval_and_one_joint_vlm_call(monkeypatch):
    cfg = load_config().model_copy(deep=True)
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

    monkeypatch.setattr("modules.prediction.get_client", lambda _cfg: client)
    monkeypatch.setattr("modules.experiments.helper.get_client", lambda _cfg: client)
    monkeypatch.setattr(
        "modules.experiments.helper.get_embedding_provider",
        lambda _cfg: FakeEmbeddingProvider(cfg.retrieval.embedding.dim),
    )
    monkeypatch.setattr(ExperienceIndex, "retrieve_objects", counted_retrieve)

    detailed = Pipeline(cfg, "e4").fit(train).predict_detailed(
        _query_with_image(test, cfg)
    )

    assert client.generation_calls == 1
    assert retrieval_calls == 1
    assert captured["schema"] is JointGripperPrediction
    assert captured["instruction"].startswith(cfg.prompts.experiments["e4"].strip())
    assert cfg.prompts.target_instructions["joint"].strip() in captured["instruction"]
    paired_payload = captured["extra"]["retrieved_objects"]
    assert len(paired_payload) == cfg.retrieval.k
    assert all("gecko_min_force_n" in item for item in paired_payload)
    assert all("silicone_min_force_n" in item for item in paired_payload)
    assert all("image_path" not in item for item in paired_payload)
    assert "retrieved_experiences" not in captured["extra"]
    assert detailed.selection.desired_gripper == "gecko"
    assert detailed.selection.model_recommended_gripper == "silicone"
    assert detailed.selection.recommendation_agrees_with_selector is False


def test_e2_and_e4_differ_only_by_experiential_retrieval_inputs(monkeypatch):
    """E4 adds paired experience retrieval to the same measured VLM query."""
    cfg = load_config().model_copy(deep=True)
    records = fabricate_records(cfg, 20)
    held = records[0].object_id
    train = [record for record in records if record.object_id != held]
    test = [record for record in records if record.object_id == held]
    query = _query_with_image(test, cfg)
    payloads = []

    class CapturingClient:
        def generate_json(self, **kwargs):
            payloads.append(kwargs["extra"])
            return JointGripperPrediction(
                gecko=PerGripperPrediction(
                    candidate_gripper=Gripper.GECKO,
                    predicted_normal_force_n=1.0,
                ),
                silicone=PerGripperPrediction(
                    candidate_gripper=Gripper.SILICONE,
                    predicted_normal_force_n=1.2,
                ),
                recommended_gripper="gecko",
            ).model_dump(mode="json")

        def cache_stats(self):
            return {}

    client = CapturingClient()
    monkeypatch.setattr("modules.prediction.get_client", lambda _cfg: client)
    monkeypatch.setattr("modules.experiments.helper.get_client", lambda _cfg: client)
    monkeypatch.setattr(
        "modules.experiments.helper.get_embedding_provider",
        lambda _cfg: FakeEmbeddingProvider(cfg.retrieval.embedding.dim),
    )
    e2 = Pipeline(cfg, "e2").fit(train).predict_detailed(query)
    e4 = Pipeline(cfg, "e4").fit(train).predict_detailed(query)

    assert e2.physics_estimates == e4.physics_estimates == {}
    assert len(payloads) == 2
    e2_payload, e4_payload = payloads
    assert e4_payload["query"] == e2_payload["query"]
    assert e4_payload["force_constraints"] == e2_payload["force_constraints"]
    assert e4_payload["roughness_measurement"] == e2_payload["roughness_measurement"]
    assert set(e4_payload) - set(e2_payload) == {
        "query_semantic_description",
        "retrieved_objects",
        "retrieval_config",
    }
    assert len(e4_payload["retrieved_objects"]) == cfg.retrieval.k


def test_e4_binary_roughness_is_internal_to_retrieval_and_categorical_for_vlm(
    monkeypatch,
):
    cfg = load_config().model_copy(deep=True)
    cfg.inputs.roughness_representation = "binary"
    cfg.inputs.use_projected_contact = False
    records = fabricate_records(cfg, 12)
    held = records[0].object_id
    train = [record for record in records if record.object_id != held]
    test = [record for record in records if record.object_id == held]
    captured = {}

    class CapturingClient:
        def generate_json(self, **kwargs):
            captured.update(kwargs["extra"])
            return JointGripperPrediction(
                gecko=PerGripperPrediction(
                    candidate_gripper=Gripper.GECKO,
                    predicted_normal_force_n=1.0,
                ),
                silicone=PerGripperPrediction(
                    candidate_gripper=Gripper.SILICONE,
                    predicted_normal_force_n=1.2,
                ),
                recommended_gripper="gecko",
            ).model_dump(mode="json")

        def cache_stats(self):
            return {}

    client = CapturingClient()
    monkeypatch.setattr("modules.prediction.get_client", lambda _cfg: client)
    monkeypatch.setattr("modules.experiments.helper.get_client", lambda _cfg: client)
    monkeypatch.setattr(
        "modules.experiments.helper.get_embedding_provider",
        lambda _cfg: FakeEmbeddingProvider(cfg.retrieval.embedding.dim),
    )

    detailed = Pipeline(cfg, "e4").fit(train).predict_detailed(
        _query_with_image(test, cfg)
    )

    assert captured["query"]["roughness_category"] in {"smooth", "rough"}
    assert "roughness_index" not in captured["query"]
    assert captured["roughness_measurement"]["numeric_values_withheld"] is True
    assert captured["retrieval_config"] == {
        "mode": "hybrid",
        "k": cfg.retrieval.k,
        "vlm_roughness_representation": "binary_category",
        "ranking_note": (
            "Neighbors were ranked upstream with the configured hybrid retrieval. "
            "Continuous roughness values are intentionally withheld; do not infer them."
        ),
    }
    for surface in captured["retrieved_objects"]:
        assert "score" not in surface
        for condition in surface["conditions"]:
            assert condition["roughness_category"] in {"smooth", "rough"}
            assert "roughness_index" not in condition
            assert "score" not in condition
            similarity = condition["similarity"]
            assert "roughness" not in similarity
            assert "roughness_contribution" not in similarity
            assert "total" not in similarity
    assert all(item.roughness_index is not None for item in detailed.retrieved_objects)
    assert all(
        item.similarity.roughness is not None for item in detailed.retrieved_objects
    )


def test_e4_optional_input_ablation_removes_roughness_and_contact(monkeypatch):
    cfg = load_config().model_copy(deep=True)
    cfg.inputs.use_roughness = False
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
    monkeypatch.setattr("modules.prediction.get_client", lambda _cfg: client)
    monkeypatch.setattr("modules.experiments.helper.get_client", lambda _cfg: client)
    monkeypatch.setattr(
        "modules.experiments.helper.get_embedding_provider",
        lambda _cfg: FakeEmbeddingProvider(cfg.retrieval.embedding.dim),
    )
    train = [record for record in records if record.object_id != held]
    test = [record for record in records if record.object_id == held]

    Pipeline(cfg, "e4").fit(train).predict_detailed(_query_with_image(test, cfg))

    assert "projected_contact_fraction" not in captured["extra"]["query"]
    assert "roughness_index" not in captured["extra"]["query"]
    assert all(
        "projected_contact_fraction" not in item
        for item in captured["extra"]["retrieved_objects"]
    )
    assert all(
        "roughness_index" not in item
        and "roughness" not in item["similarity"]
        and "contact" not in item["similarity"]
        for item in captured["extra"]["retrieved_objects"]
    )
    weights = captured["extra"]["retrieval_config"]["normalized_weights"]
    assert weights["roughness"] == 0
    assert weights["contact"] == 0
    assert captured["extra"]["retrieval_config"]["k"] == cfg.retrieval.k


@pytest.mark.parametrize("experiment", ["e1", "e3"])
def test_sensor_free_experiments_accept_missing_physical_fields(experiment, monkeypatch):
    cfg = load_config().model_copy(deep=True)
    install_gemini_fakes(monkeypatch, cfg.retrieval.embedding.dim)
    records = [
        record.model_copy(
            update={
                "mass_g": None,
                "roughness_index": None,
                "projected_contact_fraction": None,
            }
        )
        for record in fabricate_records(cfg, 12)
    ]
    held = records[0].object_id
    train = [record for record in records if record.object_id != held]
    test = [record for record in records if record.object_id == held]

    detailed = Pipeline(cfg, experiment).fit(train).predict_detailed(
        _query_with_image(test, cfg)
    )

    assert not {"mass", "roughness", "projected_contact"} & set(
        detailed.effective_inputs
    )


def test_e3_is_sensor_free_semantic_only_retrieval(monkeypatch):
    cfg = load_config().model_copy(deep=True)
    records = fabricate_records(cfg, 20)
    held = records[0].object_id
    train = [record for record in records if record.object_id != held]
    test = [record for record in records if record.object_id == held]
    captured = {}

    class CapturingClient:
        def generate_json(self, **kwargs):
            captured.update(kwargs["extra"])
            return JointGripperPrediction(
                gecko=PerGripperPrediction(
                    candidate_gripper=Gripper.GECKO, predicted_normal_force_n=1.0
                ),
                silicone=PerGripperPrediction(
                    candidate_gripper=Gripper.SILICONE, predicted_normal_force_n=1.2
                ),
                recommended_gripper="gecko",
            ).model_dump(mode="json")

        def cache_stats(self):
            return {}

    def sensors_must_not_score(*_args, **_kwargs):
        raise AssertionError("E3 must not evaluate sensor similarity")

    client = CapturingClient()
    monkeypatch.setattr("modules.prediction.get_client", lambda _cfg: client)
    monkeypatch.setattr("modules.experiments.helper.get_client", lambda _cfg: client)
    monkeypatch.setattr(
        "modules.experiments.helper.get_embedding_provider",
        lambda _cfg: FakeEmbeddingProvider(cfg.retrieval.embedding.dim),
    )
    monkeypatch.setattr("modules.retrieval.s_mass", sensors_must_not_score)
    monkeypatch.setattr("modules.retrieval.s_roughness", sensors_must_not_score)
    monkeypatch.setattr("modules.retrieval.s_contact", sensors_must_not_score)

    detailed = Pipeline(cfg, "e3").fit(train).predict_detailed(
        _query_with_image(test, cfg)
    )

    assert detailed.retrieval_mode == RetrievalMode.SEMANTIC_ONLY.value
    assert captured["query"] == {}
    assert "roughness_scale" not in captured
    assert captured["retrieval_config"] == {
        "mode": "semantic_only",
        "k": cfg.retrieval.k,
        "score": "cosine_semantic_embedding_only",
    }
    assert captured["query_semantic_description"] == detailed.semantic_description
    forbidden = {
        "mass_g",
        "roughness_index",
        "projected_contact_fraction",
        "mass",
        "roughness",
        "contact",
        "sigma_mass",
        "sigma_contact",
        "normalized_weights",
    }
    assert all(not (set(item) & forbidden) for item in captured["retrieved_objects"])
