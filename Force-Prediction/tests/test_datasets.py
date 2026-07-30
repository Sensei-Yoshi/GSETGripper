from __future__ import annotations

import base64
import hashlib
import json

import pytest

from modules.cache import DiskCache
from modules.config import load_config
from modules.contracts import Gripper
from modules.datasets import (
    DatasetObjectEdit,
    PreparationStage,
    discover_datasets,
    get_dataset,
    prepare_dataset_stages,
    update_dataset_object,
)
from modules.experiments import experiment_eligibility
from modules.perception import Description
from tests.fakes import FakeEmbeddingProvider


def _config_at(tmp_path):
    cfg = load_config().model_copy(deep=True)
    cfg.root = tmp_path
    return cfg


def test_catalog_discovers_direct_data_folders_and_excludes_artifacts(tmp_path) -> None:
    cfg = _config_at(tmp_path)
    data = tmp_path / "data"
    (data / "cache").mkdir(parents=True)
    (data / ".hidden").mkdir()
    (data / "EmptyDataset").mkdir()
    photos = data / "Photos"
    photos.mkdir()
    (photos / "object.png").write_bytes(b"source")
    (photos / "object_mask.png").write_bytes(b"derived")
    (photos / "contact_fraction" / "capture").mkdir(parents=True)
    (photos / "contact_fraction" / "capture" / "capture.png").write_bytes(b"output")

    catalog = discover_datasets(cfg)

    assert [dataset.dataset_id for dataset in catalog] == ["EmptyDataset", "Photos"]
    photos_dataset = catalog[1]
    assert list(photos_dataset.objects) == ["object"]
    item = photos_dataset.objects["object"]
    assert item.image.path == "data/Photos/object.png"
    assert item.description is None
    assert item.embedding is None
    assert item.mass_g is None
    assert item.roughness_index is None
    assert item.projected_contact_fraction is None
    assert item.split == "train"
    assert not item.gripper_outcomes
    assert photos_dataset.default_active_grippers() == (
        Gripper.GECKO,
        Gripper.SILICONE,
    )
    assert photos_dataset.runtime_config(cfg).path("cache") == (
        tmp_path / "data/cache/Photos"
    )


def test_legacy_roughness_class_is_not_converted_to_an_index(tmp_path) -> None:
    cfg = _config_at(tmp_path)
    root = tmp_path / "data/Legacy"
    root.mkdir(parents=True)
    (root / "dataset.csv").write_text(
        "Object,Image,Mass_g,roughness_class,projected_contact_fraction,"
        "silicone_force_n,silicone_feasible,gecko_force_n,gecko_feasible,"
        "favored_gripper\nCup,cup.png,100,4,0.7,1.4,True,1.1,True,gecko\n",
        encoding="utf-8",
    )

    dataset = get_dataset(cfg, "Legacy")
    item = dataset.objects["cup"]

    assert item.roughness_index is None
    assert item.legacy_roughness_class == 4
    run_cfg = dataset.runtime_config(cfg)
    assert experiment_eligibility(dataset, run_cfg, "e2").query_ids == ()
    assert "roughness index not recorded" in experiment_eligibility(
        dataset, run_cfg, "e2"
    ).query_reasons("cup")


def test_single_gripper_dataset_capabilities_and_eligibility(tmp_path) -> None:
    cfg = _config_at(tmp_path)
    root = tmp_path / "data/GeckoOnly/objects"
    image_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
        "AScY42YAAAAASUVORK5CYII="
    )
    for object_id in ("one", "two"):
        image = root / object_id / "image.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(image_bytes)
    dataset = get_dataset(cfg, "GeckoOnly")
    for index, object_id in enumerate(("one", "two"), start=1):
        dataset = update_dataset_object(
            cfg,
            dataset,
            object_id,
            DatasetObjectEdit(
                mass_g=100 + index,
                roughness_index=2,
                projected_contact_fraction=0.7,
                gecko_feasible=True,
                gecko_force_n=1.0 + index / 10,
            ),
        )

    assert dataset.capabilities.complete_gecko_labels == 2
    assert dataset.capabilities.complete_silicone_labels == 0
    assert dataset.capabilities.complete_pair_count == 0
    assert dataset.default_active_grippers() == (Gripper.GECKO,)

    run_cfg = dataset.runtime_config(cfg)
    run_cfg.prediction.active_grippers = (Gripper.GECKO,)
    report = experiment_eligibility(dataset, run_cfg, "e3")
    assert report.reference_ids == ("one", "two")
    assert report.benchmark_ids == ("one", "two")

    run_cfg.prediction.active_grippers = (Gripper.GECKO, Gripper.SILICONE)
    paired = experiment_eligibility(dataset, run_cfg, "e3")
    assert paired.reference_ids == ()
    assert paired.benchmark_ids == ()


def test_experiment_references_are_train_only_and_surface_safe(tmp_path) -> None:
    cfg = _config_at(tmp_path)
    root = tmp_path / "data/SplitReferences/objects"
    image_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
        "AScY42YAAAAASUVORK5CYII="
    )
    object_ids = ("shared", "shared__condition_2", "held_out")
    for object_id in object_ids:
        image = root / object_id / "image.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(image_bytes)

    dataset = get_dataset(cfg, "SplitReferences")
    for index, object_id in enumerate(object_ids, start=1):
        dataset = update_dataset_object(
            cfg,
            dataset,
            object_id,
            DatasetObjectEdit(
                split="test" if object_id == "held_out" else "train",
                mass_g=100 + index,
                roughness_index=2,
                projected_contact_fraction=0.7,
                gecko_feasible=True,
                gecko_force_n=1.0 + index / 10,
            ),
        )

    run_cfg = dataset.runtime_config(cfg)
    run_cfg.prediction.active_grippers = (Gripper.GECKO,)
    report = experiment_eligibility(dataset, run_cfg, "e4")

    assert report.reference_ids == ("shared", "shared__condition_2")
    assert report.query_ids == ("held_out",)
    for object_id in ("shared", "shared__condition_2"):
        assert report.query_reasons(object_id) == (
            "no eligible reference object remains after query exclusion",
        )


def test_experiment_eligibility_uses_only_enabled_inputs(tmp_path) -> None:
    cfg = _config_at(tmp_path)
    root = tmp_path / "data/Partial/objects"
    image_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
        "AScY42YAAAAASUVORK5CYII="
    )
    for object_id in ("complete", "mass_only", "image_only"):
        image = root / object_id / "image.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(image_bytes)
    dataset = get_dataset(cfg, "Partial")
    dataset = update_dataset_object(
        cfg,
        dataset,
        "complete",
        DatasetObjectEdit(
            mass_g=100,
            roughness_index=2,
            projected_contact_fraction=0.7,
            gecko_feasible=True,
            gecko_force_n=1.0,
            silicone_feasible=True,
            silicone_force_n=1.5,
        ),
    )
    dataset = update_dataset_object(
        cfg,
        dataset,
        "mass_only",
        DatasetObjectEdit(
            mass_g=120,
            gecko_feasible=True,
            gecko_force_n=1.2,
            silicone_feasible=True,
            silicone_force_n=1.7,
        ),
    )
    run_cfg = dataset.runtime_config(cfg)

    assert experiment_eligibility(dataset, run_cfg, "e1").query_ids == (
        "complete",
        "image_only",
        "mass_only",
    )
    assert experiment_eligibility(dataset, run_cfg, "e3").benchmark_ids == (
        "complete",
        "mass_only",
    )
    assert experiment_eligibility(dataset, run_cfg, "e2").query_ids == ("complete",)

    run_cfg.inputs.use_roughness = False
    run_cfg.inputs.use_projected_contact = False
    assert experiment_eligibility(dataset, run_cfg, "e2").query_ids == ("complete",)
    assert experiment_eligibility(dataset, run_cfg, "e4").query_ids == (
        "complete",
        "mass_only",
    )
    assert experiment_eligibility(dataset, run_cfg, "e5").query_ids == ()
    assert experiment_eligibility(dataset, run_cfg, "e6").query_ids == ()
    assert experiment_eligibility(dataset, run_cfg, "e4").reference_ids == (
        "complete",
        "mass_only",
    )


def test_canonical_object_folder_discovers_contact_summary(tmp_path) -> None:
    cfg = _config_at(tmp_path)
    object_dir = tmp_path / "data/Physical/objects/test_cup"
    contact_dir = object_dir / "contact_fraction"
    contact_dir.mkdir(parents=True)
    (object_dir / "image.png").write_bytes(b"source")
    (object_dir / "image_2.png").write_bytes(b"second view")
    (contact_dir / "summary.json").write_text(json.dumps({
        "schema_version": 2,
        "results": {
            "object_height_mm": 82.5,
            "object_width_mm": 47.25,
            "grasp_feasible": True,
            "antipodal_grasp": True,
            "geometric_contact_fraction": 0.72,
            "contact_floor_applied": False,
            "combined_contact_fraction": 0.72,
        },
    }))

    dataset = get_dataset(cfg, "Physical")
    item = dataset.objects["test_cup"]

    assert item.image.path == "data/Physical/objects/test_cup/image.png"
    assert item.image_2 is not None
    assert item.image_2.path == "data/Physical/objects/test_cup/image_2.png"
    assert dataset.capabilities.can_estimate_surface_area
    assert item.projected_contact_fraction == 0.72
    assert item.contact_fraction is not None
    assert item.contact_fraction.object_height_mm == 82.5
    assert item.contact_fraction.object_width_mm == 47.25
    assert item.contact_fraction.summary_path.endswith(
        "objects/test_cup/contact_fraction/summary.json"
    )


def test_flat_objectname_2_files_are_one_object_with_two_views(tmp_path) -> None:
    cfg = _config_at(tmp_path)
    root = tmp_path / "data/Photos"
    root.mkdir(parents=True)
    (root / "cup.png").write_bytes(b"primary")
    (root / "cup_2.png").write_bytes(b"second")

    dataset = get_dataset(cfg, "Photos")

    assert list(dataset.objects) == ["cup"]
    assert dataset.objects["cup"].image_2 is not None
    assert dataset.objects["cup"].image_2.path == "data/Photos/cup_2.png"
    assert dataset.capabilities.has_second_images
    assert dataset.capabilities.can_estimate_surface_area


def test_surface_area_stage_is_rejected_when_any_second_view_is_missing(tmp_path) -> None:
    cfg = _config_at(tmp_path)
    root = tmp_path / "data/Photos"
    root.mkdir(parents=True)
    (root / "cup.png").write_bytes(b"primary")
    (root / "cup_2.png").write_bytes(b"second")
    (root / "bowl.png").write_bytes(b"primary only")
    dataset = get_dataset(cfg, "Photos")

    assert not dataset.capabilities.can_estimate_surface_area
    with pytest.raises(ValueError, match="image_2 for every object"):
        prepare_dataset_stages(
            cfg,
            dataset,
            [PreparationStage.SURFACE_AREA],
        )


def test_surface_area_stage_uses_image_2_and_reuses_its_checkpoint(
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _config_at(tmp_path)
    object_dir = tmp_path / "data/Physical/objects/cup"
    object_dir.mkdir(parents=True)
    (object_dir / "image.png").write_bytes(b"primary")
    (object_dir / "image_2.png").write_bytes(b"calibrated second view")
    dataset = get_dataset(cfg, "Physical")
    calls = []

    monkeypatch.setattr(
        "modules.contact_model.create_rembg_session",
        lambda: "shared-rembg-session",
    )

    def fake_analyze(image_path, run_dir, name, params, session):
        calls.append((image_path, session))
        return None, {
            "schema_version": 2,
            "metric": "projected_two_pad_contact_fraction",
            "image": image_path.name,
            "px_per_mm": params.px_per_mm,
            "params": {
                "pad_length_mm": params.pad_length_mm,
                "minimum_bend_radius_mm": params.minimum_bend_radius_mm,
                "side_angle_deg": params.side_angle_deg,
                "minimum_contact_fraction": params.minimum_contact_fraction,
                "closing_axis": params.closing_axis,
            },
            "results": {
                "object_height_mm": 100.0,
                "object_width_mm": 40.0,
                "geometric_contact_fraction": 0.6,
                "combined_contact_fraction": 0.6,
                "grasp_feasible": True,
                "antipodal_grasp": True,
                "contact_floor_applied": False,
            },
        }, {}

    monkeypatch.setattr("modules.contact_model.analyze_image", fake_analyze)
    result = prepare_dataset_stages(
        cfg,
        dataset,
        [PreparationStage.SURFACE_AREA],
    )

    assert calls == [(object_dir / "image_2.png", "shared-rembg-session")]
    assert result["surface_area_completed"] == 1
    assert dataset.objects["cup"].projected_contact_fraction == 0.6
    summary = json.loads((object_dir / "contact_fraction/summary.json").read_text())
    assert summary["source"]["view"] == "image_2"
    assert summary["source"]["path"].endswith("objects/cup/image_2.png")

    monkeypatch.setattr(
        "modules.contact_model.analyze_image",
        lambda *args, **kwargs: pytest.fail("reusable surface result was recomputed"),
    )
    prepare_dataset_stages(
        cfg,
        dataset,
        [PreparationStage.SURFACE_AREA],
    )


def test_roughness_stage_runs_marigold_on_primary_image_and_reuses_result(
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _config_at(tmp_path)
    root = tmp_path / "data/Photos"
    root.mkdir(parents=True)
    # Valid 1x1 RGB PNG so the production image-open/hash path is exercised.
    (root / "cup.png").write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
        "AScY42YAAAAASUVORK5CYII="
    ))
    dataset = get_dataset(cfg, "Photos")
    calls = []

    class FakeAnalyzer:
        model_id = "test/marigold"

        def __init__(self, *, device, processing_resolution):
            self.device = device
            self.processing_resolution = processing_resolution

    monkeypatch.setattr("modules.models.marigold_rough.available_device", lambda: "cpu")
    monkeypatch.setattr("modules.models.marigold_rough.MarigoldAnalyzer", FakeAnalyzer)

    def fake_run(analyzer, image, output_root, **kwargs):
        calls.append(kwargs["source_path"])
        run_dir = output_root / "test-run"
        run_dir.mkdir(parents=True)
        metadata = {
            "schema_version": 3,
            "run_id": "test-run",
            "created_at": "2026-07-26T00:00:00+00:00",
            "source": {
                "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            },
            "model": {
                "id": analyzer.model_id,
                "processing_resolution": analyzer.processing_resolution,
                "num_inference_steps": kwargs["num_inference_steps"],
                "ensemble_size": kwargs["ensemble_size"],
                "seed": kwargs["seed"],
            },
            "crop": {"padding_ratio": kwargs["crop_padding_ratio"]},
            "scoring": {
                "contact_band_fraction": kwargs["contact_band_fraction"],
                "mask_erosion_ratio": kwargs["mask_erosion_ratio"],
            },
            "quality": {"status": "ok", "warnings": []},
            "roughness": {
                "mean": 0.4,
                "median": 0.35,
                "std": 0.1,
                "p25": 0.3,
                "p75": 0.45,
            },
            "roughness_uncertainty": {"mean": 0.02},
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata))
        return {**metadata, "run_dir": str(run_dir)}

    monkeypatch.setattr("modules.models.marigold_rough.run_marigold", fake_run)
    result = prepare_dataset_stages(
        cfg,
        dataset,
        [PreparationStage.ROUGHNESS],
    )

    assert calls == ["data/Photos/cup.png"]
    assert result["roughness_completed"] == 1
    assert dataset.objects["cup"].roughness is not None
    assert dataset.objects["cup"].roughness.mean == 0.4

    monkeypatch.setattr(
        "modules.models.marigold_rough.run_marigold",
        lambda *args, **kwargs: pytest.fail("reusable roughness result was recomputed"),
    )
    prepare_dataset_stages(
        cfg,
        dataset,
        [PreparationStage.ROUGHNESS],
    )


def test_description_stage_stops_without_running_downstream_stages(
    tmp_path, monkeypatch
) -> None:
    cfg = _config_at(tmp_path)
    root = tmp_path / "data/Photos"
    root.mkdir(parents=True)
    (root / "cup.png").write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
        "AScY42YAAAAASUVORK5CYII="
    ))
    dataset = get_dataset(cfg, "Photos")
    monkeypatch.setattr(
        "modules.datasets.preparation.describe",
        lambda _image, _cfg: Description(retrieval_description="Gemini cup"),
    )

    result = prepare_dataset_stages(
        cfg,
        dataset,
        [PreparationStage.DESCRIPTIONS],
    )

    checkpoint = json.loads((root / "objects/cup/descriptor.json").read_text())
    manifest = json.loads((root / "preparation_manifest.json").read_text())
    assert result["descriptors_completed"] == 1
    assert result["embeddings_completed"] == 0
    assert set(manifest["stages"]) == {"index", "descriptions"}
    assert checkpoint["descriptor_source"] == "live_gemini"
    assert checkpoint["embedding_status"] == "pending"
    assert not (tmp_path / "data/cache/Photos/experiences.jsonl").exists()
    assert get_dataset(cfg, "Photos").objects["cup"].description is not None


def test_description_stage_requires_a_source_image(tmp_path) -> None:
    cfg = _config_at(tmp_path)
    root = tmp_path / "data/Photos"
    root.mkdir(parents=True)
    image = root / "missing.png"
    image.write_bytes(b"discoverable placeholder")
    dataset = get_dataset(cfg, "Photos")
    image.unlink()

    with pytest.raises(FileNotFoundError, match="no source image"):
        prepare_dataset_stages(cfg, dataset, [PreparationStage.DESCRIPTIONS])


def test_embedding_stage_adds_description_prerequisite_and_uses_gemini_metadata(
    tmp_path, monkeypatch
) -> None:
    cfg = _config_at(tmp_path)
    root = tmp_path / "data/Photos"
    root.mkdir(parents=True)
    (root / "cup.png").write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
        "AScY42YAAAAASUVORK5CYII="
    ))
    dataset = get_dataset(cfg, "Photos")
    monkeypatch.setattr(
        "modules.datasets.preparation.describe",
        lambda _image, _cfg: Description(retrieval_description="Gemini cup"),
    )
    monkeypatch.setattr(
        "modules.datasets.preparation.get_embedding_provider",
        lambda _cfg: FakeEmbeddingProvider(cfg.retrieval.embedding.dim),
    )

    prepare_dataset_stages(
        cfg,
        dataset,
        [PreparationStage.EMBEDDINGS],
    )

    checkpoint = json.loads((root / "objects/cup/descriptor.json").read_text())
    manifest = json.loads((root / "preparation_manifest.json").read_text())
    assert set(manifest["stages"]) == {"index", "descriptions", "embeddings"}
    assert checkpoint["embedding_status"] == "ready"
    assert checkpoint["embedding_model"] == cfg.retrieval.embedding.model
    assert checkpoint["embedding_cache_key"]
    assert not (tmp_path / "data/cache/Photos/experiences.jsonl").exists()


def test_dataset_cache_namespaces_are_isolated_with_expforce_legacy_readthrough(
    tmp_path,
) -> None:
    legacy = DiskCache(tmp_path / "data/cache")
    key = DiskCache.key("embed", "model", 3, "text", None)
    legacy.put(key, [1.0, 2.0, 3.0])

    expforce = DiskCache(tmp_path / "data/cache/expforce/embeddings", legacy_root=legacy.root)
    other = DiskCache(tmp_path / "data/cache/Photos/embeddings")

    assert expforce.get(key) == [1.0, 2.0, 3.0]
    assert expforce.stats()["legacy_hits"] == 1
    assert (expforce.root / f"{key}.json").is_file()
    assert (legacy.root / f"{key}.json").is_file()
    assert other.get(key) is None
