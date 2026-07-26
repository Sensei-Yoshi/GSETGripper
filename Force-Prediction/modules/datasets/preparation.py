"""Independent, resumable preparation stages for any discovered dataset."""

from __future__ import annotations

import hashlib
import json
import tempfile
import urllib.request
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path

import cv2

from ..config import Config
from ..perception import Description, describe
from ..retrieval import build_embedding_text, get_embedding_provider
from .models import (
    Dataset,
    PreparationManifest,
    PreparationStage,
    PreparedObjectCheckpoint,
    StageStatus,
)
from .storage import (
    build_dataset_experiences,
    checkpoint_path,
    load_checkpoints,
    load_manifest,
    save_manifest,
    sha256_bytes,
    write_json_atomic,
)

Progress = Callable[[int, int, str], None]


def prepare_dataset_stages(
    cfg: Config,
    dataset: Dataset,
    stages: Iterable[PreparationStage | str],
    *,
    live: bool = True,
    progress: Progress | None = None,
) -> dict:
    selected = {PreparationStage(stage) for stage in stages}
    selected.add(PreparationStage.INDEX)
    if PreparationStage.EMBEDDINGS in selected:
        selected.add(PreparationStage.DESCRIPTIONS)
    if PreparationStage.EXPERIENCES in selected:
        selected.add(PreparationStage.DESCRIPTIONS)

    run_cfg = dataset.runtime_config(cfg)
    run_cfg.models.dry_run = not live
    manifest = load_manifest(dataset)
    if manifest.source_fingerprint != dataset.source_fingerprint:
        manifest = PreparationManifest(
            dataset_id=dataset.dataset_id,
            source_fingerprint=dataset.source_fingerprint,
        )
    checkpoints = load_checkpoints(dataset)
    ordered = [
        PreparationStage.INDEX,
        PreparationStage.DESCRIPTIONS,
        PreparationStage.EMBEDDINGS,
        PreparationStage.EXPERIENCES,
    ]
    tasks = _task_count(dataset, selected)
    completed = 0

    def report(label: str) -> None:
        nonlocal completed
        completed += 1
        if progress is not None:
            progress(completed, max(tasks, 1), label)

    for stage in ordered:
        if stage not in selected:
            continue
        status = StageStatus(status="running", total=_stage_total(dataset, stage))
        manifest.stages[stage.value] = status
        save_manifest(dataset, manifest)
        active_object: str | None = None
        try:
            if stage is PreparationStage.INDEX:
                missing = [
                    item.object_id
                    for item in dataset.objects.values()
                    if not (cfg.root / item.image.path).is_file()
                ]
                manifest.missing_images = missing
                status.completed = len(dataset.objects) - len(missing)
                report(f"indexed {dataset.display_name}")
            elif stage is PreparationStage.DESCRIPTIONS:
                for item in dataset.objects.values():
                    active_object = item.object_id
                    available = _ensure_image(cfg, item) if live else (
                        cfg.root / item.image.path
                    ).is_file()
                    checkpoint = _prepare_description(
                        run_cfg,
                        dataset,
                        item,
                        checkpoints,
                        live,
                        image_available=available,
                    )
                    checkpoints[item.object_id] = checkpoint
                    status.completed += 1
                    save_manifest(dataset, manifest)
                    report(f"description: {item.name}")
            elif stage is PreparationStage.EMBEDDINGS:
                provider = get_embedding_provider(run_cfg)
                embedding_model = (
                    run_cfg.retrieval.embedding.model
                    if live
                    else f"mock-sha256-{run_cfg.retrieval.embedding.dim}"
                )
                for item in dataset.objects.values():
                    active_object = item.object_id
                    embedding_checkpoint = checkpoints.get(item.object_id)
                    if embedding_checkpoint is None:
                        continue
                    text = build_embedding_text(embedding_checkpoint.descriptor.description)
                    descriptor_hash = sha256_bytes(text.encode("utf-8"))
                    reusable = (
                        embedding_checkpoint.embedding_status == "ready"
                        and embedding_checkpoint.embedding_model == embedding_model
                        and embedding_checkpoint.embedding_dim == run_cfg.retrieval.embedding.dim
                        and embedding_checkpoint.embedding_descriptor_sha256 == descriptor_hash
                    )
                    if not reusable:
                        vector = provider.embed(text, is_query=False)
                        embedding_checkpoint.embedding_status = "ready"
                        embedding_checkpoint.embedding_model = embedding_model
                        embedding_checkpoint.embedding_dim = len(vector)
                        embedding_checkpoint.embedding_descriptor_sha256 = descriptor_hash
                        embedding_checkpoint.embedding_sha256 = hashlib.sha256(
                            vector.tobytes()
                        ).hexdigest()
                        embedding_checkpoint.embedding_cache_key = (
                            _embedding_cache_key(run_cfg, text) if live else None
                        )
                        embedding_checkpoint.updated_at = datetime.now(UTC).isoformat()
                        write_json_atomic(
                            checkpoint_path(dataset, item.object_id),
                            embedding_checkpoint.model_dump(mode="json"),
                        )
                    status.completed += 1
                    save_manifest(dataset, manifest)
                    report(f"embedding: {item.name}")
            elif stage is PreparationStage.EXPERIENCES:
                build_dataset_experiences(dataset, run_cfg.force.limit_n)
                status.completed = len(dataset.objects)
                report(f"experience records: {dataset.display_name}")
        except Exception as error:
            status.status = "failed"
            status.failed_object = active_object
            status.error = f"{type(error).__name__}: {error}"
            status.updated_at = datetime.now(UTC).isoformat()
            save_manifest(dataset, manifest)
            raise
        status.status = "complete"
        status.updated_at = datetime.now(UTC).isoformat()
        save_manifest(dataset, manifest)

    return _legacy_manifest_view(dataset, manifest)


def _prepare_description(
    cfg: Config,
    dataset: Dataset,
    item,
    checkpoints: dict[str, PreparedObjectCheckpoint],
    live: bool,
    *,
    image_available: bool,
) -> PreparedObjectCheckpoint:
    image_path = cfg.root / item.image.path
    image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest() if image_available else None
    signature = _descriptor_signature(cfg)
    existing = checkpoints.get(item.object_id)
    reusable = bool(
        existing
        and existing.descriptor_source == "live_gemini"
        and existing.descriptor_signature == signature
        and existing.image_sha256 == image_hash
    )
    if reusable and existing is not None:
        return existing
    if live and not image_available:
        raise FileNotFoundError(f"no source image is available for {item.name!r}")
    if live:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"could not decode {image_path}")
        value = describe(image, cfg)
        source = "live_gemini"
    elif existing is not None and existing.descriptor_source == "live_gemini":
        value = existing.descriptor
        source = existing.descriptor_source
    else:
        value = _fallback_description(item.name, "offline object-name fallback")
        source = "object_name_fallback"
    checkpoint = PreparedObjectCheckpoint(
        dataset_id=dataset.dataset_id,
        object_id=item.object_id,
        object_name=item.name,
        image_name=Path(item.image.path).name,
        image_path=item.image.path,
        image_sha256=image_hash,
        descriptor_source=source,
        descriptor_model=cfg.models.vlm if source == "live_gemini" else None,
        descriptor_signature=signature,
        descriptor=value,
        updated_at=datetime.now(UTC).isoformat(),
    )
    write_json_atomic(
        checkpoint_path(dataset, item.object_id), checkpoint.model_dump(mode="json")
    )
    return checkpoint


def _ensure_image(cfg: Config, item) -> bool:
    path = cfg.root / item.image.path
    if path.is_file():
        return True
    if not item.image.remote_url:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as fh:
        temporary = Path(fh.name)
    try:
        urllib.request.urlretrieve(item.image.remote_url, temporary)
    except Exception:  # noqa: BLE001
        temporary.unlink(missing_ok=True)
        return False
    temporary.replace(path)
    return True


def _descriptor_signature(cfg: Config) -> str:
    payload = {
        "model": cfg.models.vlm,
        "system": cfg.prompts.descriptor_system,
        "instruction": cfg.prompts.descriptor,
        "schema": Description.model_json_schema(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _embedding_cache_key(cfg: Config, text: str) -> str:
    from ..cache import DiskCache

    formatted = f"title: none | text: {text}"
    return DiskCache.key(
        "embed", cfg.retrieval.embedding.model, cfg.retrieval.embedding.dim, formatted, None
    )


def _fallback_description(name: str, reason: str) -> Description:
    return Description(
        retrieval_description=name,
        contact_region="centered lateral grasp band",
        contact_material="unknown",
        visible_surface_material="unknown",
        visible_surface_condition="unknown",
        local_geometry="unknown",
        contact_patch_visibility=reason,
        uncertainty="No Gemini image descriptor is available.",
    )


def _stage_total(dataset: Dataset, stage: PreparationStage) -> int:
    del stage
    return len(dataset.objects)


def _task_count(dataset: Dataset, stages: set[PreparationStage]) -> int:
    return sum(
        1 if stage in {PreparationStage.INDEX, PreparationStage.EXPERIENCES}
        else len(dataset.objects)
        for stage in stages
    )


def _legacy_manifest_view(dataset: Dataset, manifest: PreparationManifest) -> dict:
    descriptions = manifest.stages.get(PreparationStage.DESCRIPTIONS.value, StageStatus())
    embeddings = manifest.stages.get(PreparationStage.EMBEDDINGS.value, StageStatus())
    experiences = manifest.stages.get(PreparationStage.EXPERIENCES.value, StageStatus())
    failed = next((item for item in manifest.stages.values() if item.status == "failed"), None)
    return {
        **manifest.model_dump(mode="json"),
        "status": "failed" if failed else "complete",
        "objects": len(dataset.objects),
        "experience_rows": sum(len(item.gripper_outcomes) for item in dataset.objects.values()),
        "descriptors_completed": descriptions.completed,
        "embeddings_completed": embeddings.completed,
        "records_completed": experiences.completed,
        "failed_object": failed.failed_object if failed else None,
        "error": failed.error if failed else None,
    }
