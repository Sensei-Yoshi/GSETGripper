"""Immutable Exp-Force validation adapter, split, preparation, and benchmark tools."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field, model_validator

from .config import Config
from .contracts import ExperienceRecord, Gripper, Meta, group_by_object, save_experiences
from .evaluation import EvalRow, compute_metrics
from .perception import describe
from .pipeline import Pipeline, PipelineRunResult, QueryInput

BASE_URL = "https://raw.githubusercontent.com/expforcesubmission/Exp-Force-Website/main"
SPLIT_ALGORITHM = "balanced-marginals-v2"
SOURCE_RELATIVE = Path("data/expforce/dataset_2gripper.csv")
SPLIT_RELATIVE = Path("data/expforce/validation_split.json")
EXPERIENCES_RELATIVE = Path("data/expforce/validation_experiences.jsonl")
PREPARATION_RELATIVE = Path("data/expforce/preparation_manifest.json")
RESULTS_RELATIVE = Path("data/expforce/results")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"expected True/False, got {value!r}")
    return normalized == "true"


class ExpForceRow(BaseModel):
    object_name: str
    image_name: str
    mass_g: float = Field(gt=0)
    roughness_class: int = Field(ge=1, le=5)
    projected_contact_fraction: float = Field(ge=0, le=1)
    silicone_force_n: float | None
    silicone_feasible: bool
    gecko_force_n: float | None
    gecko_feasible: bool
    favored_gripper: str

    @property
    def object_id(self) -> str:
        return slug(self.object_name)

    @model_validator(mode="after")
    def _validate_labels(self) -> ExpForceRow:
        for force, feasible, name in (
            (self.silicone_force_n, self.silicone_feasible, "silicone"),
            (self.gecko_force_n, self.gecko_feasible, "gecko"),
        ):
            if feasible and (force is None or force <= 0):
                raise ValueError(f"{name} feasible row requires positive force")
            if not feasible and force is not None:
                raise ValueError(f"{name} infeasible row must not contain force")
        expected = self.expected_favored()
        if self.favored_gripper != expected:
            raise ValueError(
                f"favored_gripper={self.favored_gripper!r}, expected {expected!r}"
            )
        return self

    def expected_favored(self) -> str:
        candidates: dict[str, float] = {}
        if self.silicone_feasible and self.silicone_force_n is not None:
            candidates["silicone"] = self.silicone_force_n
        if self.gecko_feasible and self.gecko_force_n is not None:
            candidates["gecko"] = self.gecko_force_n
        if not candidates:
            return "none"
        minimum = min(candidates.values())
        winners = [name for name, force in candidates.items() if force == minimum]
        return winners[0] if len(winners) == 1 else "tie"


def source_path(cfg: Config) -> Path:
    return cfg.root / SOURCE_RELATIVE


def source_sha256(cfg: Config) -> str:
    return hashlib.sha256(source_path(cfg).read_bytes()).hexdigest()


def load_rows(cfg: Config) -> list[ExpForceRow]:
    path = source_path(cfg)
    with path.open(newline="", encoding="utf-8") as fh:
        raw_rows = list(csv.DictReader(fh))
    rows = [
        ExpForceRow(
            object_name=row["Object"],
            image_name=row["Image"],
            mass_g=float(row["Mass_g"]),
            roughness_class=int(row["roughness_class"]),
            projected_contact_fraction=float(row["projected_contact_fraction"]),
            silicone_force_n=float(row["silicone_force_n"]) if row["silicone_force_n"] else None,
            silicone_feasible=_parse_bool(row["silicone_feasible"]),
            gecko_force_n=float(row["gecko_force_n"]) if row["gecko_force_n"] else None,
            gecko_feasible=_parse_bool(row["gecko_feasible"]),
            favored_gripper=row["favored_gripper"].strip().lower(),
        )
        for row in raw_rows
    ]
    ids = [row.object_id for row in rows]
    if len(rows) != 129:
        raise ValueError(f"expected 129 objects, found {len(rows)}")
    if len(set(ids)) != len(ids):
        raise ValueError("object names do not produce unique IDs")
    return rows


def validation_summary(cfg: Config, rows: list[ExpForceRow] | None = None) -> dict:
    rows = rows or load_rows(cfg)
    increment = cfg.force.increment_n
    off_grid = []
    for row in rows:
        for force in (row.silicone_force_n, row.gecko_force_n):
            if force is not None and not math.isclose(force / increment, round(force / increment)):
                off_grid.append(row.object_id)
    return {
        "objects": len(rows),
        "experience_rows": 2 * len(rows),
        "source_sha256": source_sha256(cfg),
        "roughness_counts": dict(sorted(Counter(row.roughness_class for row in rows).items())),
        "favored_counts": dict(sorted(Counter(row.favored_gripper for row in rows).items())),
        "off_force_grid": sorted(set(off_grid)),
    }


def to_experiences(
    cfg: Config,
    rows: list[ExpForceRow],
    descriptions: dict[str, str] | None = None,
) -> list[ExperienceRecord]:
    descriptions = descriptions or {}
    records: list[ExperienceRecord] = []
    for row in rows:
        image_path = f"data/expforce/images/{row.image_name}"
        description = descriptions.get(row.object_id, row.object_name)
        for gripper, force, feasible in (
            (Gripper.SILICONE, row.silicone_force_n, row.silicone_feasible),
            (Gripper.GECKO, row.gecko_force_n, row.gecko_feasible),
        ):
            records.append(
                ExperienceRecord(
                    object_id=row.object_id,
                    image_path=image_path,
                    mass_g=row.mass_g,
                    roughness_class=row.roughness_class,
                    projected_contact_fraction=row.projected_contact_fraction,
                    gripper=gripper,
                    min_force_n=force if feasible else None,
                    feasible=feasible,
                    failed_at_limit_n=None if feasible else cfg.force.limit_n,
                    semantic_description=description,
                    meta=Meta(pad_id="synthetic-2gripper"),
                )
            )
    return records


def _bin(value: float, edges: list[float]) -> int:
    return int(np.digitize([value], edges, right=True)[0])


def _tokens(rows: list[ExpForceRow]) -> dict[str, set[str]]:
    log_masses = np.log([row.mass_g for row in rows])
    mass_edges = list(np.quantile(log_masses, [0.25, 0.5, 0.75]))
    contacts = [row.projected_contact_fraction for row in rows]
    contact_edges = list(np.quantile(contacts, [0.25, 0.5, 0.75]))
    output: dict[str, set[str]] = {}
    for row in rows:
        feasible_forces = [
            force
            for force, feasible in (
                (row.silicone_force_n, row.silicone_feasible),
                (row.gecko_force_n, row.gecko_feasible),
            )
            if feasible and force is not None
        ]
        minimum_force = min(feasible_forces) if feasible_forces else 999.0
        output[row.object_id] = {
            f"favored:{row.favored_gripper}",
            f"roughness:{row.roughness_class}",
            f"mass:{_bin(math.log(row.mass_g), mass_edges)}",
            f"contact:{_bin(row.projected_contact_fraction, contact_edges)}",
            f"force:{_bin(minimum_force, [0.5, 1.5, 3.0])}",
        }
    return output


def make_validation_split(rows: list[ExpForceRow], seed: int, test_size: int = 29) -> dict:
    if test_size <= 0 or test_size >= len(rows):
        raise ValueError("test_size must leave non-empty reference and test sets")
    token_map = _tokens(rows)
    full = Counter(
        token
        for object_id in sorted(token_map)
        for token in sorted(token_map[object_id])
    )
    targets = {token: count * test_size / len(rows) for token, count in full.items()}
    rng = np.random.default_rng(seed)
    tie_order = {row.object_id: float(rng.random()) for row in rows}
    selected: list[str] = []
    selected_counts: Counter[str] = Counter()

    def objective(counts: Counter[str]) -> float:
        return sum(
            ((counts[token] - targets[token]) ** 2) / max(targets[token], 1.0)
            for token in sorted(targets)
        )

    remaining = {row.object_id for row in rows}
    while len(selected) < test_size:
        choices = []
        for object_id in remaining:
            candidate_counts = selected_counts.copy()
            candidate_counts.update(sorted(token_map[object_id]))
            choices.append((objective(candidate_counts), tie_order[object_id], object_id))
        _, _, chosen = min(choices)
        selected.append(chosen)
        selected_counts.update(sorted(token_map[chosen]))
        remaining.remove(chosen)

    test_ids = sorted(selected)
    reference_ids = sorted(remaining)
    return {
        "schema_version": 1,
        "algorithm": SPLIT_ALGORITHM,
        "seed": seed,
        "reference_object_ids": reference_ids,
        "test_object_ids": test_ids,
        "distribution": {
            "full": dict(sorted(full.items())),
            "test": dict(sorted(selected_counts.items())),
        },
    }


def get_or_create_split(cfg: Config, refresh: bool = False) -> dict:
    path = cfg.root / SPLIT_RELATIVE
    digest = source_sha256(cfg)
    if path.exists() and not refresh:
        split = json.loads(path.read_text())
        if split.get("source_sha256") == digest and split.get("algorithm") == SPLIT_ALGORITHM:
            return split
        raise ValueError("validation split is stale; refresh it explicitly")
    split = make_validation_split(load_rows(cfg), cfg.seed)
    split["source_file"] = str(SOURCE_RELATIVE)
    split["source_sha256"] = digest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2) + "\n")
    return split


def _download_image(image_name: str, destination: Path) -> bool:
    if destination.exists():
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(f"{BASE_URL}/images/{image_name}", destination)
    except Exception:  # noqa: BLE001 - caller reports every missing image
        return False
    return True


def prepare_dataset(
    cfg: Config,
    *,
    live: bool,
    refresh_split: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    rows = load_rows(cfg)
    split = get_or_create_split(cfg, refresh=refresh_split)
    descriptions: dict[str, str] = {}
    missing_images: list[str] = []
    image_root = cfg.root / "data/expforce/images"
    run_cfg = cfg.model_copy(deep=True)
    run_cfg.models.dry_run = not live

    for index, row in enumerate(rows, start=1):
        destination = image_root / row.image_name
        available = destination.exists() or (live and _download_image(row.image_name, destination))
        if not available:
            missing_images.append(row.image_name)
            descriptions[row.object_id] = row.object_name
        elif live:
            import cv2

            image = cv2.imread(str(destination))
            descriptions[row.object_id] = describe(image, run_cfg).description
        else:
            descriptions[row.object_id] = row.object_name
        if progress is not None:
            progress(index, len(rows), row.object_name)

    records = to_experiences(cfg, rows, descriptions)
    save_experiences(cfg.root / EXPERIENCES_RELATIVE, records)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "source_sha256": source_sha256(cfg),
        "split_sha256": hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest(),
        "descriptor_mode": "live_gemini" if live else "object_name_fallback",
        "objects": len(rows),
        "experience_rows": len(records),
        "missing_images": missing_images,
    }
    manifest_path = cfg.root / PREPARATION_RELATIVE
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_validation_records(cfg: Config) -> list[ExperienceRecord]:
    path = cfg.root / EXPERIENCES_RELATIVE
    if path.exists():
        from .contracts import load_experiences

        return load_experiences(path)
    return to_experiences(cfg, load_rows(cfg))


def load_image(cfg: Config, record: ExperienceRecord):  # noqa: ANN201
    path = cfg.root / record.image_path
    if not path.exists():
        return None
    import cv2

    return cv2.imread(str(path))


@dataclass
class BenchmarkResult:
    metrics: dict
    rows: list[dict]
    run_metadata: dict


def _retrieval_payload(detailed: PipelineRunResult) -> dict:
    return {
        gripper: [item.model_dump(mode="json") for item in items]
        for gripper, items in detailed.retrieved.items()
    }


def run_benchmark(
    cfg: Config,
    experiment: str,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> BenchmarkResult:
    split = get_or_create_split(cfg)
    records = load_validation_records(cfg)
    reference_ids = set(split["reference_object_ids"])
    test_ids = split["test_object_ids"]
    train = [record for record in records if record.object_id in reference_ids]
    by_object = group_by_object(records)
    pipe = Pipeline(cfg, cfg.experiment(experiment)).fit(train)
    eval_rows: list[EvalRow] = []
    output_rows: list[dict] = []

    for index, object_id in enumerate(test_ids, start=1):
        truth = by_object[object_id]
        sample = truth.gecko or truth.silicone
        assert sample is not None
        query = QueryInput(
            object_id=object_id,
            mass_g=sample.mass_g,
            roughness_class=sample.roughness_class,
            projected_contact_fraction=sample.projected_contact_fraction,
            image_bgr=load_image(cfg, sample),
            image_path=sample.image_path,
            semantic_description=sample.semantic_description,
        )
        detailed = pipe.predict_detailed(query)
        result = detailed.selection
        eval_rows.append(EvalRow(object_id=object_id, truth=truth, result=result))
        optimal, oracle_force = truth.optimal_grippers()
        chosen = result.desired_gripper
        chosen_truth = truth.get(Gripper(chosen)) if chosen in {"gecko", "silicone"} else None
        regret = None
        if oracle_force is not None and chosen_truth and chosen_truth.feasible:
            regret = (chosen_truth.min_force_n or 0.0) - oracle_force
        row = {
            "object_id": object_id,
            "mass_g": sample.mass_g,
            "roughness_class": sample.roughness_class,
            "projected_contact_fraction": sample.projected_contact_fraction,
            "true_gecko_force_n": truth.gecko.min_force_n if truth.gecko else None,
            "pred_gecko_force_n": result.candidate_predictions["gecko"].predicted_normal_force_n,
            "true_silicone_force_n": truth.silicone.min_force_n if truth.silicone else None,
            "pred_silicone_force_n": result.candidate_predictions["silicone"].predicted_normal_force_n,
            "true_favored": "tie" if len(optimal) > 1 else next(iter(optimal)).value,
            "predicted_gripper": chosen,
            "selection_correct": chosen in {gripper.value for gripper in optimal},
            "regret_n": regret,
            "semantic_description": detailed.semantic_description,
            "retrieval": _retrieval_payload(detailed),
            "physics_estimates": detailed.physics_estimates,
            "cache_stats": detailed.cache_stats,
        }
        output_rows.append(row)
        if progress is not None:
            progress(index, len(test_ids), object_id)

    metrics = compute_metrics(eval_rows, cfg).to_dict()
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": experiment,
        "dry_run": cfg.models.dry_run,
        "source_sha256": source_sha256(cfg),
        "split_algorithm": split["algorithm"],
        "split_seed": split["seed"],
        "model": cfg.models.vlm,
        "embedding_model": cfg.retrieval.embedding.model,
        "embedding_dim": cfg.retrieval.embedding.dim,
        "retrieval": cfg.retrieval.model_dump(mode="json"),
    }
    return BenchmarkResult(metrics=metrics, rows=output_rows, run_metadata=metadata)


def save_benchmark(cfg: Config, benchmark: BenchmarkResult) -> tuple[Path, Path]:
    output_dir = cfg.root / RESULTS_RELATIVE
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{timestamp}_{benchmark.run_metadata['experiment']}"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    json_path.write_text(
        json.dumps(
            {
                "metadata": benchmark.run_metadata,
                "metrics": benchmark.metrics,
                "rows": benchmark.rows,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    flat_fields = [key for key in benchmark.rows[0] if key not in {"retrieval", "cache_stats"}]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=flat_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(benchmark.rows)
    return json_path, csv_path
