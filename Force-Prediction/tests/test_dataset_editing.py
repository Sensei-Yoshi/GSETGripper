from __future__ import annotations

import base64
import csv
import json

import pytest

from modules.config import load_config
from modules.contracts import Gripper, load_experiences
from modules.datasets import (
    DatasetObjectEdit,
    PreparationStage,
    add_dataset_condition,
    delete_dataset_condition,
    get_dataset,
    prepare_dataset_stages,
    update_csv_dataset_object,
    update_dataset_object,
)
from modules.expforce import load_rows
from modules.perception import Description


def _config_at(tmp_path):
    cfg = load_config().model_copy(deep=True)
    cfg.root = tmp_path
    cfg.dataset_id = "Physical"
    return cfg


def _write_source(root) -> None:
    path = root / "dataset.csv"
    path.parent.mkdir(parents=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=(
                "Object",
                "Image",
                "split",
                "Mass_g",
                "roughness_index",
                "projected_contact_fraction",
                "silicone_force_n",
                "silicone_feasible",
                "gecko_force_n",
                "gecko_feasible",
                "favored_gripper",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "Object": "Test cup",
                "Image": "test.png",
                "split": "train",
                "Mass_g": 100,
                "roughness_index": 2,
                "projected_contact_fraction": 0.5,
                "silicone_force_n": 1.5,
                "silicone_feasible": True,
                "gecko_force_n": 1.0,
                "gecko_feasible": True,
                "favored_gripper": "gecko",
            }
        )


def test_csv_object_edit_refreshes_measurements_and_experiences_only(
    tmp_path, monkeypatch
) -> None:
    cfg = _config_at(tmp_path)
    root = tmp_path / "data/Physical"
    _write_source(root)
    object_dir = root / "objects/test_cup"
    object_dir.mkdir(parents=True)
    (object_dir / "image.png").write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
        "AScY42YAAAAASUVORK5CYII="
    ))
    dataset = get_dataset(cfg, "Physical")
    monkeypatch.setattr(
        "modules.datasets.preparation.describe",
        lambda _image, _cfg: Description(retrieval_description="Gemini test cup"),
    )
    prepare_dataset_stages(
        cfg,
        dataset,
        [PreparationStage.DESCRIPTIONS],
    )
    checkpoint_path = object_dir / "descriptor.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint.update(
        embedding_status="ready",
        embedding_model="test-embedding",
        embedding_dim=3,
        embedding_sha256="abc",
    )
    checkpoint_path.write_text(json.dumps(checkpoint))
    dataset = get_dataset(cfg, "Physical")

    refreshed = update_csv_dataset_object(
        cfg,
        dataset,
        "test_cup",
        DatasetObjectEdit(
            split="test",
            mass_g=125,
            roughness_index=4,
            projected_contact_fraction=0.75,
            silicone_force_n=0.8,
            silicone_feasible=True,
            gecko_force_n=1.2,
            gecko_feasible=True,
        ),
    )

    row = load_rows(cfg)[0]
    assert row.object_name == "Test cup"
    assert row.object_id == "test_cup"
    assert row.image_name == "test.png"
    assert row.split == "test"
    assert row.mass_g == 125
    assert row.favored_gripper == "silicone"
    assert object_dir.exists()
    edited_checkpoint = json.loads(checkpoint_path.read_text())
    assert edited_checkpoint["descriptor_source"] == "live_gemini"
    assert edited_checkpoint["embedding_status"] == "ready"
    assert edited_checkpoint["embedding_model"] == "test-embedding"
    assert refreshed.objects["test_cup"].mass_g == 125
    assert refreshed.objects["test_cup"].split == "test"

    experiences = load_experiences(refreshed.paths.experiences)
    assert len(experiences) == 2
    assert {record.gripper for record in experiences} == {
        Gripper.GECKO,
        Gripper.SILICONE,
    }
    assert all(record.mass_g == 125 for record in experiences)
    assert all(
        record.semantic_description == "Gemini test cup"
        for record in experiences
    )
    manifest = json.loads(refreshed.paths.preparation_manifest.read_text())
    assert manifest["source_fingerprint"] == refreshed.source_fingerprint
    assert manifest["stages"]["experiences"]["status"] == "complete"


def test_image_folder_partial_edit_is_persisted_and_builds_only_completed_outcomes(
    tmp_path,
) -> None:
    cfg = _config_at(tmp_path)
    cfg.dataset_id = "Photos"
    image = tmp_path / "data/Photos/objects/cup/image.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
        "AScY42YAAAAASUVORK5CYII="
    ))
    dataset = get_dataset(cfg, "Photos")
    original_fingerprint = dataset.source_fingerprint

    refreshed = update_dataset_object(
        cfg,
        dataset,
        "cup",
        DatasetObjectEdit(
            split="test",
            mass_g=None,
            roughness_index=None,
            projected_contact_fraction=None,
            gecko_feasible=True,
            gecko_force_n=1.25,
            silicone_feasible=True,
            silicone_force_n=None,
        ),
    )

    measurement_path = image.parent / "measurements.json"
    persisted = json.loads(measurement_path.read_text())
    assert persisted["schema_version"] == 2
    assert persisted["split"] == "test"
    assert persisted["mass_g"] is None
    assert refreshed.objects["cup"].split == "test"
    assert refreshed.source_fingerprint != original_fingerprint
    assert refreshed.objects["cup"].gripper_outcomes[Gripper.GECKO].complete
    assert not refreshed.objects["cup"].gripper_outcomes[Gripper.SILICONE].complete
    experiences = load_experiences(refreshed.paths.experiences)
    assert len(experiences) == 1
    assert experiences[0].gripper is Gripper.GECKO
    assert experiences[0].mass_g is None


def test_csv_edit_accepts_blank_measurements_and_unrecorded_outcomes(tmp_path) -> None:
    cfg = _config_at(tmp_path)
    root = tmp_path / "data/Physical"
    _write_source(root)
    dataset = get_dataset(cfg, "Physical")

    update_dataset_object(
        cfg,
        dataset,
        "test_cup",
        DatasetObjectEdit(),
    )

    row = load_rows(cfg)[0]
    assert row.mass_g is None
    assert row.roughness_index is None
    assert row.projected_contact_fraction is None
    assert row.gecko_feasible is None
    assert row.silicone_feasible is None
    assert row.favored_gripper is None


def test_csv_surface_supports_unlimited_shared_artifact_conditions(
    tmp_path, monkeypatch
) -> None:
    cfg = _config_at(tmp_path)
    root = tmp_path / "data/Physical"
    _write_source(root)
    object_dir = root / "objects/test_cup"
    object_dir.mkdir(parents=True)
    (object_dir / "image.png").write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
        "AScY42YAAAAASUVORK5CYII="
    ))
    dataset = get_dataset(cfg, "Physical")
    monkeypatch.setattr(
        "modules.datasets.preparation.describe",
        lambda _image, _cfg: Description(retrieval_description="shared cup surface"),
    )
    prepare_dataset_stages(cfg, dataset, [PreparationStage.DESCRIPTIONS])
    dataset = get_dataset(cfg, "Physical")

    created_ids = []
    for index in range(2, 5):
        dataset, object_id = add_dataset_condition(
            cfg,
            dataset,
            "test_cup",
            DatasetObjectEdit(
                mass_g=100 + index,
                roughness_index=2 + index,
                projected_contact_fraction=0.5 + index / 100,
                gecko_feasible=True,
                gecko_force_n=1 + index / 10,
            ),
        )
        created_ids.append(object_id)

    assert created_ids == [
        "test_cup__condition_2",
        "test_cup__condition_3",
        "test_cup__condition_4",
    ]
    assert dataset.summary()["physical_surfaces"] == 1
    assert dataset.summary()["measurement_conditions"] == 4
    assert dataset.summary()["unique_photos"] == 1
    assert len(dataset.descriptions) == 1
    assert all(item.description is not None for item in dataset.objects.values())
    assert {item.split for item in dataset.objects.values()} == {"train"}
    assert len(load_experiences(dataset.paths.experiences)) == 5

    with pytest.raises(ValueError, match="identical"):
        add_dataset_condition(
            cfg,
            dataset,
            "test_cup",
            DatasetObjectEdit(
                mass_g=102,
                roughness_index=4,
                projected_contact_fraction=0.52,
            ),
        )
    with pytest.raises(ValueError, match="baseline"):
        delete_dataset_condition(cfg, dataset, "test_cup")
    dataset = delete_dataset_condition(cfg, dataset, created_ids[1])
    assert created_ids[1] not in dataset.objects


def test_condition_requirements_follow_enabled_measurement_modes(tmp_path) -> None:
    cfg = _config_at(tmp_path)
    root = tmp_path / "data/Physical"
    _write_source(root)
    dataset = get_dataset(cfg, "Physical")
    with pytest.raises(ValueError, match="contact fraction"):
        add_dataset_condition(
            cfg,
            dataset,
            "test_cup",
            DatasetObjectEdit(mass_g=120, roughness_index=3),
        )

    cfg.inputs.use_projected_contact = False
    dataset, object_id = add_dataset_condition(
        cfg,
        dataset,
        "test_cup",
        DatasetObjectEdit(mass_g=120, roughness_index=3),
    )
    assert dataset.objects[object_id].projected_contact_fraction is None
