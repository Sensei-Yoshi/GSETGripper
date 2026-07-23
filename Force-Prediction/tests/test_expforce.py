from __future__ import annotations

import shutil

import cv2
import numpy as np
import pytest

from force_prediction import expforce
from force_prediction.config import load_config
from force_prediction.contracts import CandidateQuery, Gripper, PerGripperPrediction
from force_prediction.expforce import (
    load_experience_pool,
    load_rows,
    prepare_dataset,
    run_benchmark,
    source_path,
    to_experiences,
    validation_summary,
)
from force_prediction.perception import Description
from force_prediction.physics import PhysicsEstimate
from force_prediction.prediction import vlm_predict_gripper


def test_source_validation_and_paired_conversion():
    cfg = load_config().model_copy(deep=True)
    original = source_path(cfg).read_bytes()
    rows = load_rows(cfg)
    summary = validation_summary(cfg, rows)
    records = to_experiences(cfg, rows)

    assert len(rows) == 129
    assert len(records) == 258
    assert summary["favored_counts"].get("tie", 0) == 0
    assert set(summary["favored_counts"]) == {"gecko", "silicone"}
    assert {record.object_id for record in records} == {row.object_id for row in rows}
    assert all(sum(record.object_id == row.object_id for record in records) == 2 for row in rows)
    assert source_path(cfg).read_bytes() == original


def test_all_objects_form_the_experience_pool(tmp_path):
    source_cfg = load_config().model_copy(deep=True)
    cfg = source_cfg.model_copy(deep=True)
    cfg.root = tmp_path
    source = tmp_path / "data/expforce/dataset_2gripper.csv"
    source.parent.mkdir(parents=True)
    shutil.copyfile(source_path(source_cfg), source)

    manifest = prepare_dataset(cfg, live=False)
    records = load_experience_pool(cfg)

    assert manifest["status"] == "complete"
    assert manifest["descriptors_completed"] == 129
    assert manifest["embeddings_completed"] == 0
    assert len(records) == 258
    assert len({record.object_id for record in records}) == 129


def test_e5_does_not_use_or_send_physics():
    cfg = load_config().model_copy(deep=True)
    e5 = cfg.experiment("e5")

    assert e5.use_vlm is True
    assert e5.use_retrieval is True
    assert e5.use_paired_rows is True
    assert e5.use_physics is False
    assert "surface patches" in cfg.prompts.descriptor_system
    assert "text embedding" in cfg.prompts.descriptor_system


def test_each_vlm_experiment_routes_to_an_explicit_config_prompt():
    cfg = load_config()

    assert set(cfg.prompts.experiments) == {"e1", "e2", "e3", "e5"}
    for experiment in ("e1", "e2", "e3", "e5"):
        toggles = cfg.experiment(experiment)
        assert toggles.use_vlm is True
        assert toggles.prompt == experiment
        assert cfg.prompts.experiments[experiment].strip()
    for experiment in ("e3b", "e4", "e6"):
        assert cfg.experiment(experiment).prompt is None


def test_e1_payload_is_truly_zero_shot_and_uses_e1_prompt(monkeypatch):
    cfg = load_config().model_copy(deep=True)
    cfg.models.dry_run = False
    captured = {}

    class FakeClient:
        def generate_json(self, **kwargs):
            captured.update(kwargs)
            return PerGripperPrediction(
                candidate_gripper=Gripper.GECKO,
                predicted_normal_force_n=1.137,
            ).model_dump(mode="json")

    monkeypatch.setattr("force_prediction.prediction.get_client", lambda _cfg: FakeClient())
    query = CandidateQuery(
        object_id="zero_shot_query",
        image_path="",
        mass_g=321,
        roughness_class=4,
        projected_contact_fraction=0.63,
        candidate_gripper=Gripper.GECKO,
    )

    prediction = vlm_predict_gripper(
        cfg,
        query,
        None,
        [],
        None,
        include_paired=False,
        instruction=cfg.prompts.experiments["e1"],
        include_retrieval=False,
        include_measured=False,
    )

    assert captured["instruction"].startswith("E1 ZERO-SHOT VISION-ONLY CONDITION")
    assert set(captured["extra"]["query"]) == {"object_id", "candidate_gripper"}
    assert "retrieved_experiences" not in captured["extra"]
    assert "retrieval_config" not in captured["extra"]
    assert "roughness_scale" not in captured["extra"]
    assert prediction.predicted_normal_force_n == 1.137


def test_vlm_payload_never_contains_physics(monkeypatch):
    cfg = load_config().model_copy(deep=True)
    cfg.models.dry_run = False
    captured = {}

    class FakeClient:
        def generate_json(self, **kwargs):
            captured.update(kwargs["extra"])
            return PerGripperPrediction(
                candidate_gripper=Gripper.GECKO,
                predicted_normal_force_n=1.0,
            ).model_dump(mode="json")

    monkeypatch.setattr("force_prediction.prediction.get_client", lambda _cfg: FakeClient())
    query = CandidateQuery(
        object_id="query",
        image_path="",
        mass_g=100,
        roughness_class=2,
        projected_contact_fraction=0.8,
        semantic_description="smooth sidewall",
        candidate_gripper=Gripper.GECKO,
    )
    physics = PhysicsEstimate(
        gripper=Gripper.GECKO,
        feasible=True,
        min_force_n=0.5,
        raw_force_n=0.4,
    )

    vlm_predict_gripper(
        cfg,
        query,
        None,
        [],
        physics,
        include_paired=False,
        instruction=cfg.prompts.experiments["e3"],
        include_retrieval=True,
    )

    assert "physics_force_estimate_n" not in captured
    assert "physics_feasible" not in captured


def test_full_129_object_leave_one_out_benchmark_runs_offline(tmp_path):
    source_cfg = load_config().model_copy(deep=True)
    cfg = source_cfg.model_copy(deep=True)
    cfg.root = tmp_path
    cfg.models.dry_run = True
    source = tmp_path / "data/expforce/dataset_2gripper.csv"
    source.parent.mkdir(parents=True)
    shutil.copyfile(source_path(source_cfg), source)

    benchmark = run_benchmark(cfg, "e5")

    assert len(benchmark.rows) == 129
    assert benchmark.metrics["force"]["overall"]["n"] == 258
    assert benchmark.metrics["selection"]["n"] == 129
    assert benchmark.run_metadata["evaluation_protocol"] == "leave-one-object-out"
    assert benchmark.run_metadata["training_objects_per_run"] == 128


def test_live_preparation_checkpoints_and_resumes(tmp_path, monkeypatch):
    source_cfg = load_config().model_copy(deep=True)
    cfg = source_cfg.model_copy(deep=True)
    cfg.root = tmp_path
    source = tmp_path / "data/expforce/dataset_2gripper.csv"
    source.parent.mkdir(parents=True)
    shutil.copyfile(source_path(source_cfg), source)
    ok, encoded = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok

    def fake_download(_name, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(encoded.tobytes())
        return True

    calls = 0

    def flaky_describe(_image, _cfg):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("quota pause")
        return Description(retrieval_description=f"contact descriptor {calls}")

    monkeypatch.setattr(expforce, "_download_image", fake_download)
    monkeypatch.setattr(expforce, "describe", flaky_describe)

    with pytest.raises(RuntimeError, match="quota pause"):
        prepare_dataset(cfg, live=True)

    assert len(list((tmp_path / expforce.DESCRIPTORS_RELATIVE).glob("*.json"))) == 2

    resumed_calls = 0

    def resumed_describe(_image, _cfg):
        nonlocal resumed_calls
        resumed_calls += 1
        return Description(retrieval_description=f"resumed descriptor {resumed_calls}")

    class TextOnlyProvider:
        calls = 0

        def embed(self, text, image_bgr=None, is_query=False):
            assert text
            assert image_bgr is None
            assert is_query is False
            self.calls += 1
            return np.ones(cfg.retrieval.embedding.dim, dtype=np.float32)

    provider = TextOnlyProvider()
    monkeypatch.setattr(expforce, "describe", resumed_describe)
    monkeypatch.setattr(expforce, "get_embedding_provider", lambda _cfg: provider)

    manifest = prepare_dataset(cfg, live=True)

    assert resumed_calls == 127
    assert provider.calls == 129
    assert manifest["status"] == "complete"
    assert manifest["descriptors_completed"] == 129
    assert manifest["embeddings_completed"] == 129
