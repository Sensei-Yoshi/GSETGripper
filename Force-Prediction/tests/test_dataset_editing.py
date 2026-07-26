from __future__ import annotations

import csv
import json

from modules.config import load_config
from modules.contracts import Gripper, load_experiences
from modules.datasets import (
    DatasetObjectEdit,
    PreparationStage,
    get_dataset,
    prepare_dataset_stages,
    update_csv_dataset_object,
)
from modules.expforce import load_rows


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
                "Mass_g",
                "roughness_class",
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
                "Mass_g": 100,
                "roughness_class": 2,
                "projected_contact_fraction": 0.5,
                "silicone_force_n": 1.5,
                "silicone_feasible": True,
                "gecko_force_n": 1.0,
                "gecko_feasible": True,
                "favored_gripper": "gecko",
            }
        )


def test_csv_object_edit_refreshes_measurements_and_experiences_only(tmp_path) -> None:
    cfg = _config_at(tmp_path)
    root = tmp_path / "data/Physical"
    _write_source(root)
    object_dir = root / "objects/test_cup"
    object_dir.mkdir(parents=True)
    (object_dir / "image.png").write_bytes(b"image")
    dataset = get_dataset(cfg, "Physical")
    prepare_dataset_stages(
        cfg,
        dataset,
        [PreparationStage.DESCRIPTIONS],
        live=False,
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
            mass_g=125,
            roughness_class=4,
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
    assert row.mass_g == 125
    assert row.favored_gripper == "silicone"
    assert object_dir.exists()
    edited_checkpoint = json.loads(checkpoint_path.read_text())
    assert edited_checkpoint["descriptor_source"] == "object_name_fallback"
    assert edited_checkpoint["embedding_status"] == "ready"
    assert edited_checkpoint["embedding_model"] == "test-embedding"
    assert refreshed.objects["test_cup"].mass_g == 125

    experiences = load_experiences(refreshed.paths.experiences)
    assert len(experiences) == 2
    assert {record.gripper for record in experiences} == {
        Gripper.GECKO,
        Gripper.SILICONE,
    }
    assert all(record.mass_g == 125 for record in experiences)
    assert all(
        record.semantic_description == "Test cup"
        for record in experiences
    )
    manifest = json.loads(refreshed.paths.preparation_manifest.read_text())
    assert manifest["source_fingerprint"] == refreshed.source_fingerprint
    assert manifest["stages"]["experiences"]["status"] == "complete"
