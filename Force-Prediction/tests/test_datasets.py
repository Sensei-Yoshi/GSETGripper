from __future__ import annotations

import json
import shutil

from modules.cache import DiskCache
from modules.config import load_config
from modules.datasets import (
    PreparationStage,
    discover_datasets,
    get_dataset,
    prepare_dataset_stages,
)
from modules.expforce import source_path


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
    assert item.roughness_class is None
    assert item.projected_contact_fraction is None
    assert not item.gripper_outcomes
    assert photos_dataset.runtime_config(cfg).path("cache") == (
        tmp_path / "data/cache/Photos"
    )


def test_paired_csv_adapter_exposes_measurements_and_outcomes(tmp_path) -> None:
    cfg = _config_at(tmp_path)
    destination = tmp_path / "data/expforce/dataset_2gripper.csv"
    destination.parent.mkdir(parents=True)
    source_cfg = load_config().model_copy(deep=True)
    shutil.copyfile(source_path(source_cfg), destination)

    dataset = get_dataset(cfg, "expforce")
    first = next(iter(dataset.objects.values()))

    assert len(dataset.objects) == 129
    assert first.mass_g is not None
    assert first.roughness_class is not None
    assert first.projected_contact_fraction is not None
    assert set(outcome.gripper.value for outcome in first.gripper_outcomes.values()) == {
        "gecko",
        "silicone",
    }
    assert dataset.capabilities.can_run_pipeline
    assert dataset.capabilities.can_benchmark


def test_description_stage_stops_without_running_downstream_stages(tmp_path) -> None:
    cfg = _config_at(tmp_path)
    root = tmp_path / "data/Photos"
    root.mkdir(parents=True)
    (root / "cup.png").write_bytes(b"source image bytes")
    dataset = get_dataset(cfg, "Photos")

    result = prepare_dataset_stages(
        cfg,
        dataset,
        [PreparationStage.DESCRIPTIONS],
        live=False,
    )

    checkpoint = json.loads((root / "descriptors/cup.json").read_text())
    manifest = json.loads((root / "preparation_manifest.json").read_text())
    assert result["descriptors_completed"] == 1
    assert result["embeddings_completed"] == 0
    assert set(manifest["stages"]) == {"index", "descriptions"}
    assert checkpoint["descriptor_source"] == "object_name_fallback"
    assert checkpoint["embedding_status"] == "pending"
    assert not (root / "validation_experiences.jsonl").exists()
    assert get_dataset(cfg, "Photos").objects["cup"].description is not None


def test_embedding_stage_adds_description_prerequisite_and_marks_mock_source(tmp_path) -> None:
    cfg = _config_at(tmp_path)
    root = tmp_path / "data/Photos"
    root.mkdir(parents=True)
    (root / "cup.png").write_bytes(b"source image bytes")
    dataset = get_dataset(cfg, "Photos")

    prepare_dataset_stages(
        cfg,
        dataset,
        [PreparationStage.EMBEDDINGS],
        live=False,
    )

    checkpoint = json.loads((root / "descriptors/cup.json").read_text())
    manifest = json.loads((root / "preparation_manifest.json").read_text())
    assert set(manifest["stages"]) == {"index", "descriptions", "embeddings"}
    assert checkpoint["embedding_status"] == "ready"
    assert checkpoint["embedding_model"].startswith("mock-sha256-")
    assert checkpoint["embedding_cache_key"] is None
    assert not (root / "validation_experiences.jsonl").exists()


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
