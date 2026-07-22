from __future__ import annotations

import shutil

from force_prediction.config import load_config
from force_prediction.expforce import (
    load_rows,
    make_validation_split,
    run_benchmark,
    source_path,
    to_experiences,
    validation_summary,
)


def test_source_validation_and_paired_conversion():
    cfg = load_config().model_copy(deep=True)
    original = source_path(cfg).read_bytes()
    rows = load_rows(cfg)
    summary = validation_summary(cfg, rows)
    records = to_experiences(cfg, rows)

    assert len(rows) == 129
    assert len(records) == 258
    assert summary["off_force_grid"] == []
    assert {record.object_id for record in records} == {row.object_id for row in rows}
    assert all(sum(record.object_id == row.object_id for record in records) == 2 for row in rows)
    assert source_path(cfg).read_bytes() == original


def test_validation_split_is_deterministic_and_balanced():
    cfg = load_config().model_copy(deep=True)
    rows = load_rows(cfg)
    first = make_validation_split(rows, cfg.seed)
    second = make_validation_split(rows, cfg.seed)

    assert first == second
    reference = set(first["reference_object_ids"])
    test = set(first["test_object_ids"])
    assert len(reference) == 100
    assert len(test) == 29
    assert reference.isdisjoint(test)
    assert reference | test == {row.object_id for row in rows}
    test_distribution = first["distribution"]["test"]
    assert all(test_distribution[f"roughness:{value}"] > 0 for value in range(1, 6))
    assert [test_distribution[f"favored:{name}"] for name in ("gecko", "silicone", "tie")] == [11, 9, 9]


def test_full_29_object_benchmark_runs_offline(tmp_path):
    source_cfg = load_config().model_copy(deep=True)
    cfg = source_cfg.model_copy(deep=True)
    cfg.root = tmp_path
    cfg.models.dry_run = True
    source = tmp_path / "data/expforce/dataset_2gripper.csv"
    source.parent.mkdir(parents=True)
    shutil.copyfile(source_path(source_cfg), source)

    benchmark = run_benchmark(cfg, "e5")

    assert len(benchmark.rows) == 29
    assert benchmark.metrics["force"]["overall"]["n"] == 58
    assert benchmark.metrics["selection"]["n"] == 29

