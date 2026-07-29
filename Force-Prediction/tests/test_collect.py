from __future__ import annotations

import builtins
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from modules import collect
from modules.config import load_config
from modules.contracts import load_experiences
from modules.perception import Description


def _config_at(tmp_path, dataset_id: str):
    cfg = load_config().model_copy(deep=True)
    cfg.root = tmp_path
    cfg.dataset_id = dataset_id
    return cfg


def test_mock_collection_writes_canonical_dataset_without_segmentation(
    tmp_path, monkeypatch
) -> None:
    cfg = _config_at(tmp_path, "mock")
    monkeypatch.setattr(
        collect,
        "create_rembg_session",
        lambda: pytest.fail("mock collection must not load rembg"),
    )
    monkeypatch.setattr(
        collect,
        "describe",
        lambda *_args: Description(retrieval_description="test Gemini descriptor"),
    )

    collect.collect_mock(cfg, 1, coarse=cfg.collection.coarse_step_n, fine=0.25)

    dataset_csv = tmp_path / "data/mock/dataset.csv"
    rows = list(csv.DictReader(dataset_csv.open()))
    assert len(rows) == 1
    object_id = rows[0]["Object"]
    object_dir = tmp_path / "data/mock/objects" / object_id
    assert (object_dir / "image.png").is_file()
    assert not (object_dir / "contact_fraction").exists()
    records = load_experiences(tmp_path / "data/cache/mock/experiences.jsonl")
    assert len(records) == 2
    assert all(record.meta.contact_fraction_source == "synthetic" for record in records)


class _FakeGripper:
    def __init__(self, _cfg, port=None):
        self.port = port

    def close_until_contact(self):
        pass

    def set_normal_force(self, _force):
        pass

    def attempt_lift(self):
        return True

    def open(self):
        pass

    def read_n(self):
        return 1.0


class _FakeRoughness:
    def read_index(self):
        return 347.82


class _FakeMass:
    def read_g(self):
        return 125.0


class _FakeCamera:
    def capture_rgb(self):
        return np.zeros((32, 32, 3), dtype=np.uint8)


def _patch_real_hardware(monkeypatch) -> None:
    from modules import hardware

    monkeypatch.setattr(hardware, "SerialGripper", _FakeGripper)
    monkeypatch.setattr(hardware, "SerialRoughness", _FakeRoughness)
    monkeypatch.setattr(hardware, "ManualMass", _FakeMass)
    monkeypatch.setattr(hardware, "OrbbecCamera", _FakeCamera)
    monkeypatch.setattr(collect, "create_rembg_session", lambda: object())
    monkeypatch.setattr(
        collect,
        "measure_pair",
        lambda *_args: (True, 1.25, [1.25, 1.25, 1.25]),
    )
    monkeypatch.setattr(
        collect,
        "describe",
        lambda *_args: Description(retrieval_description="test object"),
    )


def _inputs(monkeypatch, values: list[str]) -> None:
    responses = iter(values)
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(responses))


def test_real_collection_integrates_contact_summary_and_provenance(
    tmp_path, monkeypatch
) -> None:
    cfg = _config_at(tmp_path, "physical")
    _patch_real_hardware(monkeypatch)

    def fake_analyze(image_path, run_dir, name, _params, session=None):
        assert image_path.name == "image.png"
        assert session is not None
        run_dir.mkdir(parents=True)
        summary = {
            "schema_version": 2,
            "name": name,
            "results": {
                "object_height_mm": 80.0,
                "object_width_mm": 45.0,
                "grasp_feasible": True,
                "antipodal_grasp": True,
                "geometric_contact_fraction": 0.7,
                "contact_floor_applied": False,
                "combined_contact_fraction": 0.7,
            },
        }
        summary_path = run_dir / "summary.json"
        summary_path.write_text(json.dumps(summary))
        estimate = SimpleNamespace(
            feasible=True,
            pair=SimpleNamespace(antipodal=True),
        )
        return estimate, summary, {"summary": summary_path}

    monkeypatch.setattr(collect, "analyze_image", fake_analyze)
    _inputs(monkeypatch, ["Test Bottle", "", "", "g-1", "", "s-1", ""])

    collect.collect_real(cfg, coarse=1.0, fine=0.25, port=None)

    object_dir = tmp_path / "data/physical/objects/test_bottle"
    assert (object_dir / "image.png").is_file()
    assert (object_dir / "contact_fraction/summary.json").is_file()
    row = next(csv.DictReader((tmp_path / "data/physical/dataset.csv").open()))
    assert row["Image"] == "objects/test_bottle/image.png"
    assert float(row["roughness_index"]) == pytest.approx(347.82)
    assert float(row["projected_contact_fraction"]) == 0.7
    records = load_experiences(tmp_path / "data/cache/physical/experiences.jsonl")
    assert len(records) == 2
    assert all(record.meta.contact_model_schema_version == 2 for record in records)
    assert all(record.meta.contact_antipodal_grasp is True for record in records)


@pytest.mark.parametrize("failure", ["segmentation", "non_antipodal"])
def test_real_collection_contact_failure_writes_no_partial_object_or_records(
    tmp_path, monkeypatch, failure
) -> None:
    cfg = _config_at(tmp_path, "physical")
    _patch_real_hardware(monkeypatch)

    def fake_analyze(_image_path, _run_dir, _name, _params, session=None):
        if failure == "segmentation":
            raise RuntimeError("no foreground")
        return (
            SimpleNamespace(feasible=False, pair=SimpleNamespace(antipodal=False)),
            {},
            {},
        )

    monkeypatch.setattr(collect, "analyze_image", fake_analyze)
    _inputs(monkeypatch, ["Bad Object", "", ""])

    collect.collect_real(cfg, coarse=1.0, fine=0.25, port=None)

    assert not (tmp_path / "data/physical/objects/bad_object").exists()
    assert not (tmp_path / "data/physical/dataset.csv").exists()
    assert not (tmp_path / "data/cache/physical/experiences.jsonl").exists()
