"""Resumable, provenance-locked E1–E4 benchmark suites."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from .config import EXPERIMENT_DEFINITION_VERSION, Config, prompt_bundle_sha256
from .expforce import (
    load_rows,
    prompt_provenance,
    run_benchmark,
    save_benchmark,
    source_sha256,
)

PRIMARY_EXPERIMENTS = ("e1", "e2", "e3", "e4")


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as fh:
        json.dump(payload, fh, indent=2, default=str)
        fh.write("\n")
        temporary = Path(fh.name)
    temporary.replace(path)


def _suite_snapshot(cfg: Config) -> dict:
    definitions = {
        name: cfg.experiment(name).model_dump(mode="json") for name in PRIMARY_EXPERIMENTS
    }
    return {
        "dataset_id": cfg.dataset_id,
        "source_sha256": source_sha256(cfg),
        "prompt_bundle_sha256": prompt_bundle_sha256(cfg),
        "experiment_definition_version": EXPERIMENT_DEFINITION_VERSION,
        "experiment_definitions": definitions,
        "models": {
            "vlm": cfg.models.vlm,
            "embedding": cfg.retrieval.embedding.model,
            "embedding_dim": cfg.retrieval.embedding.dim,
            "dry_run": cfg.models.dry_run,
        },
        "retrieval": cfg.retrieval.model_dump(mode="json"),
        "inputs": cfg.inputs.model_dump(mode="json"),
        "evaluation_protocol": "leave-one-object-out",
        "object_count": len(load_rows(cfg)),
    }


def _snapshot_hash(snapshot: dict) -> str:
    encoded = json.dumps(snapshot, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def create_suite(cfg: Config) -> dict:
    created_at = datetime.now(UTC)
    suite_id = created_at.strftime("%Y%m%dT%H%M%S%fZ_primary_e1_e4")
    snapshot = _suite_snapshot(cfg)
    manifest = {
        "schema_version": 5,
        "suite_id": suite_id,
        "created_at": created_at.isoformat(),
        "updated_at": created_at.isoformat(),
        "status": "pending",
        "execution_mode": "Offline" if cfg.models.dry_run else "Live Gemini",
        "experiments": list(PRIMARY_EXPERIMENTS),
        "snapshot": snapshot,
        "snapshot_sha256": _snapshot_hash(snapshot),
        "prompt_context": prompt_provenance(cfg, "e1"),
        "runs": {
            name: {"status": "pending", "json_path": None, "csv_path": None}
            for name in PRIMARY_EXPERIMENTS
        },
        "exports": {},
    }
    _write_json_atomic(suite_manifest_path(cfg, suite_id), manifest)
    return manifest


def suite_manifest_path(cfg: Config, suite_id: str) -> Path:
    return cfg.root / "data" / cfg.dataset_id / "suites" / suite_id / "manifest.json"


def load_suite(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def list_suites(cfg: Config) -> list[dict]:
    root = cfg.root / "data" / cfg.dataset_id / "suites"
    manifests = []
    for path in sorted(root.glob("*/manifest.json"), reverse=True):
        try:
            manifest = load_suite(path)
        except (OSError, json.JSONDecodeError):
            continue
        manifest["manifest_path"] = str(path)
        manifests.append(manifest)
    return manifests


def run_suite(
    cfg: Config,
    manifest: dict | None = None,
    *,
    progress: Callable[[str, int, int, str], None] | None = None,
) -> dict:
    """Run or resume a suite, checkpointing after every completed experiment."""
    manifest = manifest or create_suite(cfg)
    current_snapshot = _suite_snapshot(cfg)
    if _snapshot_hash(current_snapshot) != manifest["snapshot_sha256"]:
        raise ValueError(
            "Current data, prompts, model, retrieval, or experiment definitions differ "
            "from this suite snapshot; start a new suite instead of resuming it."
        )
    path = suite_manifest_path(cfg, manifest["suite_id"])
    manifest["status"] = "running"
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    _write_json_atomic(path, manifest)

    total_objects = int(manifest["snapshot"]["object_count"])
    try:
        for experiment in PRIMARY_EXPERIMENTS:
            state = manifest["runs"][experiment]
            saved = state.get("json_path")
            if state.get("status") == "completed" and saved and (cfg.root / saved).is_file():
                continue
            state["status"] = "running"
            _write_json_atomic(path, manifest)

            def on_object(
                done: int,
                total: int,
                object_id: str,
                active_experiment: str = experiment,
            ) -> None:
                if progress is not None:
                    progress(active_experiment, done, total, object_id)

            benchmark = run_benchmark(cfg, experiment, progress=on_object)
            json_path, csv_path = save_benchmark(cfg, benchmark)
            state.update(
                status="completed",
                json_path=str(json_path.relative_to(cfg.root)),
                csv_path=str(csv_path.relative_to(cfg.root)),
                completed_at=datetime.now(UTC).isoformat(),
                object_count=total_objects,
            )
            manifest["updated_at"] = datetime.now(UTC).isoformat()
            _write_json_atomic(path, manifest)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["last_error"] = f"{type(error).__name__}: {error}"
        manifest["updated_at"] = datetime.now(UTC).isoformat()
        _write_json_atomic(path, manifest)
        raise

    manifest["status"] = "completed"
    manifest.pop("last_error", None)
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    _write_json_atomic(path, manifest)
    return manifest


def suite_benchmarks(cfg: Config, manifest: dict) -> dict[str, dict]:
    artifacts: dict[str, dict] = {}
    for experiment in PRIMARY_EXPERIMENTS:
        relative = manifest["runs"][experiment].get("json_path")
        if not relative:
            continue
        path = cfg.root / relative
        if path.is_file():
            artifacts[experiment] = json.loads(path.read_text(encoding="utf-8"))
    return artifacts


def update_suite_exports(cfg: Config, manifest: dict, exports: dict[str, str]) -> dict:
    manifest["exports"] = exports
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    _write_json_atomic(suite_manifest_path(cfg, manifest["suite_id"]), manifest)
    return manifest
