from __future__ import annotations

import shutil

import cv2
import numpy as np
import pytest
import yaml

from modules.artifacts import artifact_backend_label, backend_provenance
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
from modules.contracts import Gripper, JointGripperPrediction, PerGripperPrediction, Query
from modules.datasets import PreparationStage, get_dataset, prepare_dataset_stages
from modules.experiments import EXPERIMENT_CATALOG, create_strategy
from modules.perception import Description
from modules.prediction import vlm_predict_joint
from tests.fakes import FakeEmbeddingProvider, install_gemini_fakes

UNKNOWN_EXPERIMENT_ID = "e6"


def test_config_and_catalog_expose_contiguous_active_experiments() -> None:
    cfg = load_config()

    assert EXPERIMENT_IDS == ("e1", "e2", "e3", "e4", "e5")
    assert tuple(cfg.experiments) == EXPERIMENT_IDS
    assert tuple(EXPERIMENT_CATALOG) == EXPERIMENT_IDS
    assert cfg.experiment("e1").method is ExperimentMethod.VISION_VLM
    assert cfg.experiment("e2").method is ExperimentMethod.SEMANTIC_RETRIEVAL_VLM
    for experiment in ("e3", "e4", "e5"):
        assert cfg.experiment(experiment).method is ExperimentMethod.HYBRID_RETRIEVAL_VLM
    with pytest.raises(KeyError, match="unknown experiment"):
        cfg.experiment(UNKNOWN_EXPERIMENT_ID)
    with pytest.raises(KeyError, match="unknown experiment"):
        create_strategy(cfg, UNKNOWN_EXPERIMENT_ID)


def test_active_experiment_evidence_ladder_is_exact() -> None:
    e1, e2, e3, e4, e5 = (EXPERIMENT_CATALOG[name] for name in EXPERIMENT_IDS)

    assert not e1.uses_measurements and e1.retrieval_mode is None
    assert not e2.uses_measurements and e2.retrieval_mode.value == "semantic_only"
    assert e3.visible_condition_fields == ("mass_g",)
    assert e4.visible_condition_fields == ("mass_g", "roughness_index")
    assert e5.visible_condition_fields == (
        "mass_g",
        "roughness_index",
        "projected_contact_fraction",
    )
    assert e5.ranking_features == ("semantic", "mass", "roughness")


def test_prediction_config_requires_nonempty_unique_stable_gripper_order() -> None:
    assert PredictionConfig(active_grippers=(Gripper.SILICONE,)).active_grippers == (
        Gripper.SILICONE,
    )
    with pytest.raises(ValueError, match="at least one"):
        PredictionConfig(active_grippers=())
    with pytest.raises(ValueError, match="duplicates"):
        PredictionConfig(active_grippers=(Gripper.GECKO, Gripper.GECKO))
    with pytest.raises(ValueError, match="stable order"):
        PredictionConfig(active_grippers=(Gripper.SILICONE, Gripper.GECKO))


def _write_deprecated_config(tmp_path, *, dry_run: bool = False, provider: bool = False):
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


def test_deprecated_runtime_config_fields_are_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="models.dry_run is no longer supported"):
        load_config(_write_deprecated_config(tmp_path, dry_run=True))
    with pytest.raises(ValueError, match="embedding.provider is no longer supported"):
        load_config(_write_deprecated_config(tmp_path, provider=True))


def test_backend_provenance_exposes_only_active_paths() -> None:
    cfg = load_config()

    assert backend_provenance(cfg, "e1") == {
        "force": "gemini_joint_generation",
        "semantic_embedding": None,
    }
    assert backend_provenance(cfg, "e5")["semantic_embedding"] == (
        cfg.retrieval.embedding.model
    )
    with pytest.raises(KeyError, match="unknown experiment"):
        backend_provenance(cfg, UNKNOWN_EXPERIMENT_ID)
    assert artifact_backend_label({"backend": {"force": "gemini_joint_generation"}}) == (
        "gemini_joint_generation"
    )


def test_each_experiment_routes_to_one_exact_prompt() -> None:
    cfg = load_config()

    assert tuple(cfg.prompts.experiments) == EXPERIMENT_IDS
    for experiment in EXPERIMENT_IDS:
        assert cfg.experiment(experiment).prompt == experiment
        assert cfg.prompts.experiments[experiment].strip()


def test_prediction_prompts_require_auditable_continuous_evidence() -> None:
    cfg = load_config()
    shared = " ".join(cfg.prompts.prediction_system.lower().split())
    e1_prompt = " ".join(cfg.prompts.experiments["e1"].lower().split())

    assert "evidence_used" in shared
    assert "calculation_summary" in shared
    assert "assumptions_and_uncertainty" in shared
    assert "do not invent" in shared
    assert "continuous roughness index" in shared
    assert "rough visual approximations" in e1_prompt


def test_e1_payload_is_image_only(monkeypatch) -> None:
    cfg = load_config().model_copy(deep=True)
    captured = {}

    class FakeClient:
        def generate_json(self, **kwargs):
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

    monkeypatch.setattr("modules.prediction.get_client", lambda _cfg: FakeClient())
    query = Query(
        object_id="zero_shot_query",
        image_path="",
        mass_g=321,
        roughness_index=4,
        projected_contact_fraction=0.63,
    )

    vlm_predict_joint(
        cfg,
        query,
        np.zeros((8, 8, 3), dtype=np.uint8),
        [],
        instruction=cfg.prompts.experiments["e1"],
        include_retrieval=False,
        include_measured=False,
    )

    assert captured["extra"]["query"] == {}
    assert "retrieved_objects" not in captured["extra"]
    assert "retrieval_config" not in captured["extra"]
    assert "roughness_measurement" not in captured["extra"]


def test_single_silicone_benchmark_omits_inactive_outputs(tmp_path, monkeypatch) -> None:
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
    assert all("pred_gecko_force_n" not in row for row in batch.rows)


def test_gemini_preparation_checkpoints_and_resumes(tmp_path, monkeypatch) -> None:
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

    assert len(list((root / "objects").glob("*/descriptor.json"))) == 2
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
