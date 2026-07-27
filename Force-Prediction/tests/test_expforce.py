from __future__ import annotations

import json
import shutil

import cv2
import numpy as np
import pytest
import yaml

from modules.config import EXPERIMENT_IDS, ExperimentMethod, load_config
from modules.contracts import (
    Gripper,
    JointGripperPrediction,
    PerGripperPrediction,
    Query,
)
from modules.datasets import PreparationStage, get_dataset, prepare_dataset_stages
from modules.expforce import (
    artifact_backend_label,
    backend_provenance,
    load_rows,
    prepare_dataset,
    run_benchmark,
    save_pipeline_run,
    saved_run_experiment_label,
    source_path,
    to_experiences,
    validation_summary,
)
from modules.hardware import fabricate_records
from modules.perception import Description
from modules.pipeline import Pipeline, query_input_from_object
from modules.prediction import vlm_predict_joint
from tests.fakes import FakeEmbeddingProvider, install_gemini_fakes


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


def test_expforce_preparation_wrapper_delegates_to_canonical_gemini_pipeline(monkeypatch):
    cfg = load_config().model_copy(deep=True)
    dataset = object()
    captured = {}

    monkeypatch.setattr("modules.datasets.get_dataset", lambda _cfg, _id: dataset)

    def fake_prepare(_cfg, selected, stages, *, progress=None):
        captured.update(dataset=selected, stages=stages, progress=progress)
        return {"status": "complete"}

    monkeypatch.setattr("modules.datasets.prepare_dataset_stages", fake_prepare)
    def progress(*_args):
        pass

    manifest = prepare_dataset(cfg, progress=progress)

    assert manifest == {"status": "complete"}
    assert captured["dataset"] is dataset
    assert captured["stages"] == [
        PreparationStage.DESCRIPTIONS,
        PreparationStage.EMBEDDINGS,
        PreparationStage.EXPERIENCES,
    ]
    assert captured["progress"] is progress


def test_config_exposes_only_the_final_explicit_experiment_methods():
    cfg = load_config().model_copy(deep=True)

    assert tuple(cfg.experiments) == EXPERIMENT_IDS
    assert cfg.experiment("e1").method is ExperimentMethod.JOINT_VLM
    assert cfg.experiment("e2").method is ExperimentMethod.JOINT_VLM_MEASURED
    assert cfg.experiment("e3").method is ExperimentMethod.SEMANTIC_RETRIEVAL_VLM
    assert cfg.experiment("e4").method is ExperimentMethod.PAIRED_RETRIEVAL_VLM
    assert cfg.experiment("e5").method is ExperimentMethod.CALIBRATED_PHYSICS
    assert cfg.experiment("e6").method is ExperimentMethod.PHYSICS_SEMANTIC_RESIDUAL
    assert "surface patches" in cfg.prompts.descriptor_system
    assert "text embedding" in cfg.prompts.descriptor_system
    assert not hasattr(cfg.models, "dry_run")
    assert not hasattr(cfg.retrieval.embedding, "provider")


def _write_deprecated_config(tmp_path, *, dry_run=False, provider=False):
    source_cfg = load_config()
    raw = yaml.safe_load((source_cfg.root / "config.yaml").read_text())
    if dry_run:
        raw["models"]["dry_run"] = True
    if provider:
        raw["retrieval"]["embedding"]["provider"] = "mock"
    destination = tmp_path / "config.yaml"
    destination.write_text(yaml.safe_dump(raw, sort_keys=False))
    shutil.copyfile(source_cfg.root / "prompts.yaml", tmp_path / "prompts.yaml")
    return destination


def test_deprecated_dry_run_config_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="models.dry_run is no longer supported"):
        load_config(_write_deprecated_config(tmp_path, dry_run=True))


def test_deprecated_embedding_provider_config_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="embedding.provider is no longer supported"):
        load_config(_write_deprecated_config(tmp_path, provider=True))


def test_backend_provenance_distinguishes_gemini_and_local_force_paths():
    cfg = load_config()

    assert backend_provenance(cfg, "e1") == {
        "force": "gemini_joint_generation",
        "semantic_embedding": None,
    }
    assert backend_provenance(cfg, "e5") == {
        "force": "local_calibrated_physics",
        "semantic_embedding": None,
    }
    assert backend_provenance(cfg, "e6")["semantic_embedding"] == (
        cfg.retrieval.embedding.model
    )
    assert artifact_backend_label({"execution_mode": "Offline"}) == "Legacy Offline"


def test_new_single_run_artifact_uses_backend_provenance(tmp_path):
    source_cfg = load_config().model_copy(deep=True)
    cfg = source_cfg.model_copy(deep=True)
    cfg.root = tmp_path
    source = tmp_path / "data/expforce/dataset.csv"
    source.parent.mkdir(parents=True)
    shutil.copyfile(source_path(source_cfg), source)
    records = fabricate_records(cfg, 12)
    held = records[0].object_id
    query_records = [record for record in records if record.object_id == held]
    training = [record for record in records if record.object_id != held]
    detailed = Pipeline(cfg, "e5").fit(training).predict_detailed(
        query_input_from_object(query_records, cfg)
    )

    path = save_pipeline_run(
        cfg,
        detailed=detailed,
        experiment="e5",
        query={
            "object_id": held,
            "mass_g": query_records[0].mass_g,
            "roughness_class": query_records[0].roughness_class,
            "projected_contact_fraction": query_records[0].projected_contact_fraction,
        },
        truth=None,
        counterfactual=True,
    )
    artifact = json.loads(path.read_text())

    assert artifact["schema_version"] == 6
    assert artifact["backend"] == {
        "force": "local_calibrated_physics",
        "semantic_embedding": None,
    }
    assert "execution_mode" not in artifact


def test_each_vlm_experiment_routes_to_an_explicit_config_prompt():
    cfg = load_config()

    assert set(cfg.prompts.experiments) == {"e1", "e2", "e3", "e4"}
    for experiment in ("e1", "e2", "e3", "e4"):
        definition = cfg.experiment(experiment)
        assert definition.prompt == experiment
        assert cfg.prompts.experiments[experiment].strip()
    for experiment in ("e5", "e6"):
        assert cfg.experiment(experiment).prompt is None


def test_vlm_prompts_require_auditable_evidence_without_invented_constants():
    cfg = load_config()
    shared = " ".join(cfg.prompts.prediction_system.lower().split())
    e1_prompt = " ".join(cfg.prompts.experiments["e1"].lower().split())
    e2_prompt = " ".join(cfg.prompts.experiments["e2"].lower().split())

    assert "evidence_used" in shared
    assert "calculation_summary" in shared
    assert "assumptions_and_uncertainty" in shared
    assert "do not invent" in shared
    assert "hidden chain-of-thought" not in shared
    assert "side-view" not in shared
    assert "rough visual approximations" in e1_prompt
    assert "authoritative measurements" in e2_prompt
    assert "do not infer hidden" in e2_prompt


def test_e1_payload_is_truly_zero_shot_and_uses_e1_prompt(monkeypatch):
    cfg = load_config().model_copy(deep=True)
    captured = {}

    class FakeClient:
        calls = 0

        def generate_json(self, **kwargs):
            self.calls += 1
            captured.update(kwargs)
            return JointGripperPrediction(
                gecko=PerGripperPrediction(
                    candidate_gripper=Gripper.GECKO,
                    predicted_normal_force_n=1.137,
                ),
                silicone=PerGripperPrediction(
                    candidate_gripper=Gripper.SILICONE,
                    predicted_normal_force_n=1.731,
                ),
                recommended_gripper="gecko",
            ).model_dump(mode="json")

    client = FakeClient()
    monkeypatch.setattr("modules.prediction.get_client", lambda _cfg: client)
    query = Query(
        object_id="zero_shot_query",
        image_path="",
        mass_g=321,
        roughness_class=4,
        projected_contact_fraction=0.63,
    )

    prediction = vlm_predict_joint(
        cfg,
        query,
        np.zeros((8, 8, 3), dtype=np.uint8),
        [],
        instruction=cfg.prompts.experiments["e1"],
        include_retrieval=False,
        include_measured=False,
    )

    assert client.calls == 1
    assert captured["instruction"].startswith("E1 ZERO-SHOT VISION-ONLY CONDITION")
    assert captured["extra"]["query"] == {}
    assert "retrieved_objects" not in captured["extra"]
    assert "retrieval_config" not in captured["extra"]
    assert "roughness_scale" not in captured["extra"]
    assert set(captured["extra"]["gripper_embodiments"]) == {"gecko", "silicone"}
    assert "context_images" not in captured
    assert "require_context_images" not in captured
    assert prediction.gecko.predicted_normal_force_n == 1.137
    assert prediction.silicone.predicted_normal_force_n == 1.731
    assert prediction.recommended_gripper == "gecko"


def test_e2_joint_payload_contains_measurements_but_no_retrieval_or_physics(monkeypatch):
    cfg = load_config().model_copy(deep=True)
    captured = {}

    class FakeClient:
        calls = 0

        def generate_json(self, **kwargs):
            self.calls += 1
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

    client = FakeClient()
    monkeypatch.setattr("modules.prediction.get_client", lambda _cfg: client)
    query = Query(
        object_id="query",
        image_path="",
        mass_g=100,
        roughness_class=2,
        projected_contact_fraction=0.8,
        semantic_description="smooth sidewall",
    )

    vlm_predict_joint(
        cfg,
        query,
        np.zeros((8, 8, 3), dtype=np.uint8),
        [],
        instruction=cfg.prompts.experiments["e2"],
        include_retrieval=False,
        include_measured=True,
    )

    assert client.calls == 1
    assert captured["query"]["mass_g"] == 100
    assert captured["query"]["roughness_class"] == 2
    assert captured["query"]["projected_contact_fraction"] == 0.8
    assert "retrieved_objects" not in captured
    assert "physics_force_estimate_n" not in captured
    assert "physics_feasible" not in captured


def test_legacy_e4_e5_artifact_labels_preserve_old_meanings():
    assert saved_run_experiment_label(
        {
            "experiment": "e5",
            "experiment_toggles": {
                "use_retrieval": True,
                "use_paired_rows": True,
                "use_vlm": True,
                "use_physics": False,
                "use_residual": False,
            },
        }
    ) == "Legacy E5 — paired retrieval VLM"
    assert saved_run_experiment_label(
        {
            "experiment": "e4",
            "experiment_toggles": {
                "use_retrieval": False,
                "use_paired_rows": False,
                "use_vlm": False,
                "use_physics": True,
                "use_residual": False,
            },
        }
    ) == "Legacy E4 — calibrated physics"
def test_full_129_object_leave_one_out_benchmark_uses_gemini_contract(
    tmp_path, monkeypatch
):
    source_cfg = load_config().model_copy(deep=True)
    cfg = source_cfg.model_copy(deep=True)
    cfg.root = tmp_path
    install_gemini_fakes(monkeypatch, cfg.retrieval.embedding.dim)
    monkeypatch.setattr(
        "modules.expforce.load_image",
        lambda *_args: np.zeros((8, 8, 3), dtype=np.uint8),
    )
    source = tmp_path / "data/expforce/dataset.csv"
    source.parent.mkdir(parents=True)
    shutil.copyfile(source_path(source_cfg), source)

    benchmark = run_benchmark(cfg, "e4")

    assert len(benchmark.rows) == 129
    assert benchmark.metrics["force"]["overall"]["n"] == 258
    assert benchmark.metrics["selection"]["n"] == 129
    assert benchmark.run_metadata["evaluation_protocol"] == "leave-one-object-out"
    assert benchmark.run_metadata["training_objects_per_run"] == 128
    assert benchmark.run_metadata["experiment_method"] == "paired_retrieval_vlm"
    assert benchmark.run_metadata["backend"]["force"] == "gemini_joint_generation"
    assert "dry_run" not in benchmark.run_metadata
    assert benchmark.metrics["model_recommendation"]["n"] == 129


def test_gemini_preparation_checkpoints_and_resumes(tmp_path, monkeypatch):
    cfg = load_config().model_copy(deep=True)
    cfg.root = tmp_path
    root = tmp_path / "data/Photos"
    root.mkdir(parents=True)
    ok, encoded = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    for name in ("a", "b", "c"):
        (root / f"{name}.png").write_bytes(encoded.tobytes())
    dataset = get_dataset(cfg, "Photos")

    calls = 0

    def flaky_describe(_image, _cfg):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("quota pause")
        return Description(retrieval_description=f"contact descriptor {calls}")

    monkeypatch.setattr("modules.datasets.preparation.describe", flaky_describe)

    with pytest.raises(RuntimeError, match="quota pause"):
        prepare_dataset_stages(
            cfg,
            dataset,
            [PreparationStage.DESCRIPTIONS, PreparationStage.EMBEDDINGS],
        )

    assert len(
        list((root / "objects").glob("*/descriptor.json"))
    ) == 2

    resumed_calls = 0

    def resumed_describe(_image, _cfg):
        nonlocal resumed_calls
        resumed_calls += 1
        return Description(retrieval_description=f"resumed descriptor {resumed_calls}")

    provider = FakeEmbeddingProvider(cfg.retrieval.embedding.dim)
    monkeypatch.setattr("modules.datasets.preparation.describe", resumed_describe)
    monkeypatch.setattr(
        "modules.datasets.preparation.get_embedding_provider", lambda _cfg: provider
    )

    manifest = prepare_dataset_stages(
        cfg,
        dataset,
        [PreparationStage.DESCRIPTIONS, PreparationStage.EMBEDDINGS],
    )

    assert resumed_calls == 1
    assert provider.client.embedding_calls == 3
    assert manifest["status"] == "complete"
    assert manifest["descriptors_completed"] == 3
    assert manifest["embeddings_completed"] == 3
