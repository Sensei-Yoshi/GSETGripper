from __future__ import annotations

import shutil

import cv2
import numpy as np
import pytest
import yaml

from modules.benchmarking import (
    evaluate_benchmark_predictions,
    generate_benchmark_predictions,
)
from modules.config import (
    EXPERIMENT_IDS,
    ExperimentMethod,
    PredictionConfig,
    load_config,
)
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
    prepare_dataset,
    saved_run_experiment_label,
)
from modules.perception import Description
from modules.prediction import vlm_predict_joint
from tests.fakes import FakeEmbeddingProvider, install_gemini_fakes


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
    assert cfg.experiment("e1").method is ExperimentMethod.VISION_VLM
    assert cfg.experiment("e2").method is ExperimentMethod.MEASURED_VLM
    assert cfg.experiment("e3").method is ExperimentMethod.SEMANTIC_RETRIEVAL_VLM
    assert cfg.experiment("e4").method is ExperimentMethod.HYBRID_RETRIEVAL_VLM
    with pytest.raises(KeyError, match="unknown experiment"):
        cfg.experiment("e5")
    assert "exact surface contacted by the gripper pads" in cfg.prompts.descriptor_system
    assert "Do not mention object identity" in cfg.prompts.descriptor_system
    assert "curvature" in cfg.prompts.descriptor_system
    assert "text embedding" in cfg.prompts.descriptor_system
    assert not hasattr(cfg.models, "dry_run")
    assert not hasattr(cfg.retrieval.embedding, "provider")


def test_prediction_config_requires_nonempty_unique_stable_gripper_order():
    assert PredictionConfig(active_grippers=(Gripper.SILICONE,)).active_grippers == (
        Gripper.SILICONE,
    )
    with pytest.raises(ValueError, match="at least one"):
        PredictionConfig(active_grippers=())
    with pytest.raises(ValueError, match="duplicates"):
        PredictionConfig(active_grippers=(Gripper.GECKO, Gripper.GECKO))
    with pytest.raises(ValueError, match="stable order"):
        PredictionConfig(active_grippers=(Gripper.SILICONE, Gripper.GECKO))


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


def test_backend_provenance_exposes_only_active_gemini_paths():
    cfg = load_config()

    assert backend_provenance(cfg, "e1") == {
        "force": "gemini_joint_generation",
        "semantic_embedding": None,
    }
    with pytest.raises(KeyError, match="unknown experiment"):
        backend_provenance(cfg, "e5")
    assert artifact_backend_label({"execution_mode": "Offline"}) == "Legacy Offline"


def test_each_vlm_experiment_routes_to_an_explicit_config_prompt():
    cfg = load_config()

    assert set(cfg.prompts.experiments) == {"e1", "e2", "e3", "e4"}
    for experiment in ("e1", "e2", "e3", "e4"):
        definition = cfg.experiment(experiment)
        assert definition.prompt == experiment
        assert cfg.prompts.experiments[experiment].strip()


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
        roughness_index=4,
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
        roughness_index=347.82,
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
    assert captured["query"]["roughness_index"] == pytest.approx(347.82)
    assert captured["roughness_measurement"] == {
        "metric_name": cfg.roughness.metric_name,
        "units": cfg.roughness.units,
        "higher_is_rougher": True,
        "characteristic_scale": cfg.roughness.characteristic_scale,
    }
    assert captured["query"]["projected_contact_fraction"] == 0.8
    assert "retrieved_objects" not in captured
    assert "physics_force_estimate_n" not in captured
    assert "physics_feasible" not in captured


def test_e2_binary_roughness_payload_withholds_numerical_index(monkeypatch):
    cfg = load_config().model_copy(deep=True)
    cfg.inputs.roughness_representation = "binary"
    captured = {}

    class FakeClient:
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

    monkeypatch.setattr("modules.prediction.get_client", lambda _cfg: FakeClient())
    query = Query(
        object_id="query",
        image_path="",
        mass_g=100,
        roughness_index=1340.0,
        projected_contact_fraction=0.8,
        semantic_description="rough sidewall",
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

    assert captured["query"] == {
        "mass_g": 100,
        "roughness_category": "rough",
        "projected_contact_fraction": 0.8,
    }
    assert captured["roughness_measurement"] == {
        "representation": "binary_category",
        "categories": ["smooth", "rough"],
        "scope": "relative_to_this_dataset",
        "numeric_values_withheld": True,
    }
    assert "roughness_index" not in str(captured)
    assert "1340" not in str(captured)


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
def test_single_silicone_benchmark_omits_inactive_outputs(tmp_path, monkeypatch):
    cfg = load_config().model_copy(deep=True)
    cfg.root = tmp_path
    cfg.dataset_id = "silicone_only"
    cfg.prediction.active_grippers = (Gripper.SILICONE,)
    install_gemini_fakes(monkeypatch, cfg.retrieval.embedding.dim)
    monkeypatch.setattr(
        cv2,
        "imread",
        lambda *_args, **_kwargs: np.zeros((8, 8, 3), dtype=np.uint8),
    )
    root = tmp_path / "data/silicone_only"
    root.mkdir(parents=True)
    (root / "dataset.csv").write_text(
        "Object,Image,Mass_g,roughness_index,projected_contact_fraction,"
        "silicone_force_n,silicone_feasible,gecko_force_n,gecko_feasible,favored_gripper\n"
        "Cup,cup.png,100,2,0.7,1.1,True,,,\n"
        "Box,box.png,120,3,0.6,1.4,True,,,\n",
        encoding="utf-8",
    )
    (root / "cup.png").write_bytes(b"image")
    (root / "box.png").write_bytes(b"image")

    batch = generate_benchmark_predictions(cfg, "e1")
    evaluation = evaluate_benchmark_predictions(cfg, batch)

    assert len(batch.rows) == 2
    assert set(evaluation.metrics["force"]) == {"silicone", "overall"}
    assert evaluation.metrics["selection"] == {"applicable": False, "n": 0}
    assert batch.metadata["active_grippers"] == ["silicone"]
    assert batch.metadata["generation_mode"] == "single"
    assert batch.metadata["backend"]["force"] == "gemini_single_generation"
    assert all("pred_gecko_force_n" not in row for row in batch.rows)
    assert all("pred_silicone_force_n" in row for row in batch.rows)


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
