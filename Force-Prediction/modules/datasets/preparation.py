"""Independent, resumable preparation stages for any discovered dataset."""

from __future__ import annotations

import hashlib
import json
import tempfile
import urllib.request
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2

from ..config import Config
from ..perception import Description, describe
from ..retrieval import build_embedding_text, get_embedding_provider
from .models import (
    ContactFractionArtifact,
    Dataset,
    DatasetObject,
    PreparationManifest,
    PreparationStage,
    PreparedObjectCheckpoint,
    RoughnessArtifact,
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
MARIGOLD_PROCESSING_RESOLUTION = 768
MARIGOLD_INFERENCE_STEPS = 4
MARIGOLD_ENSEMBLE_SIZE = 3
MARIGOLD_CROP_PADDING_RATIO = 0.15
MARIGOLD_CONTACT_BAND_FRACTION = 0.60
MARIGOLD_MASK_EROSION_RATIO = 0.01


def prepare_dataset_stages(
    cfg: Config,
    dataset: Dataset,
    stages: Iterable[PreparationStage | str],
    *,
    progress: Progress | None = None,
) -> dict:
    selected = {PreparationStage(stage) for stage in stages}
    selected.add(PreparationStage.INDEX)
    if PreparationStage.EMBEDDINGS in selected:
        selected.add(PreparationStage.DESCRIPTIONS)
    if PreparationStage.EXPERIENCES in selected:
        selected.add(PreparationStage.DESCRIPTIONS)
    if (
        PreparationStage.SURFACE_AREA in selected
        and not dataset.capabilities.can_estimate_surface_area
    ):
        raise ValueError(
            f"dataset {dataset.display_name!r} needs a local image_2 for every object "
            "before surface-area/contact estimation can run"
        )

    run_cfg = dataset.runtime_config(cfg)
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
        PreparationStage.ROUGHNESS,
        PreparationStage.SURFACE_AREA,
        PreparationStage.EXPERIENCES,
    ]
    tasks = _task_count(dataset, selected)
    surface_items = _surface_items(dataset)
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
                    for item in surface_items
                    if not (cfg.root / item.image.path).is_file()
                ]
                manifest.missing_images = missing
                manifest.missing_second_images = [
                    item.object_id
                    for item in surface_items
                    if item.image_2 is None or not (cfg.root / item.image_2.path).is_file()
                ]
                status.completed = len(surface_items) - len(missing)
                report(f"indexed {dataset.display_name}")
            elif stage is PreparationStage.DESCRIPTIONS:
                for item in surface_items:
                    active_object = item.object_id
                    available = _ensure_image(cfg, item)
                    checkpoint = _prepare_description(
                        run_cfg,
                        dataset,
                        item,
                        checkpoints,
                        image_available=available,
                    )
                    checkpoints[item.object_id] = checkpoint
                    status.completed += 1
                    save_manifest(dataset, manifest)
                    report(f"description: {item.name}")
            elif stage is PreparationStage.EMBEDDINGS:
                provider = get_embedding_provider(run_cfg)
                embedding_model = run_cfg.retrieval.embedding.model
                for item in surface_items:
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
                        embedding_checkpoint.embedding_cache_key = _embedding_cache_key(
                            run_cfg, text
                        )
                        embedding_checkpoint.updated_at = datetime.now(UTC).isoformat()
                        write_json_atomic(
                            checkpoint_path(dataset, item.object_id),
                            embedding_checkpoint.model_dump(mode="json"),
                        )
                    status.completed += 1
                    save_manifest(dataset, manifest)
                    report(f"embedding: {item.name}")
            elif stage is PreparationStage.ROUGHNESS:
                from ..models.background_remover import (
                    DEFAULT_BACKGROUND_MODEL,
                    BackgroundRemover,
                )
                from ..models.marigold_rough import MarigoldAnalyzer, available_device

                device = available_device()
                analyzer = MarigoldAnalyzer(
                    device=device,
                    processing_resolution=MARIGOLD_PROCESSING_RESOLUTION,
                )
                background_remover = BackgroundRemover(model_name=DEFAULT_BACKGROUND_MODEL)
                for item in surface_items:
                    active_object = item.object_id
                    _prepare_roughness(
                        cfg,
                        dataset,
                        item,
                        analyzer=analyzer,
                        background_remover=background_remover,
                    )
                    status.completed += 1
                    save_manifest(dataset, manifest)
                    report(f"Marigold roughness: {item.name}")
            elif stage is PreparationStage.SURFACE_AREA:
                from ..contact_model import ContactParams

                params = ContactParams(
                    px_per_mm=float(cfg.geometry.px_per_mm),
                    closing_axis="x",
                    mode=cfg.geometry.contact_mode,
                    pad_length_mm=float(cfg.geometry.pad_length_mm),
                    minimum_bend_radius_mm=float(cfg.geometry.minimum_bend_radius_mm),
                    side_angle_deg=float(cfg.geometry.side_angle_deg),
                    minimum_contact_fraction=float(cfg.geometry.minimum_contact_fraction),
                    rigid_contact_tolerance_mm=float(cfg.geometry.rigid_contact_tolerance_mm),
                    finger_extension_mm=float(cfg.geometry.finger_extension_mm),
                    planar_threshold=float(cfg.geometry.planar_threshold),
                )
                session = None
                for item in surface_items:
                    active_object = item.object_id
                    session = _prepare_surface_area(
                        cfg,
                        dataset,
                        item,
                        params=params,
                        session=session,
                    )
                    status.completed += 1
                    save_manifest(dataset, manifest)
                    report(f"surface/contact estimate: {item.name}")
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
    if not image_available:
        raise FileNotFoundError(f"no source image is available for {item.name!r}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"could not decode {image_path}")
    value = describe(image, cfg)
    source = "live_gemini"
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
    write_json_atomic(checkpoint_path(dataset, item.object_id), checkpoint.model_dump(mode="json"))
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


def _prepare_roughness(
    cfg: Config,
    dataset: Dataset,
    item: DatasetObject,
    *,
    analyzer: Any,
    background_remover: Any,
) -> None:
    from PIL import Image

    from ..models.marigold_rough import run_marigold

    image_path = cfg.root / item.image.path
    if not image_path.is_file():
        raise FileNotFoundError(f"no source image is available for {item.name!r}")
    with Image.open(image_path) as source:
        source_rgb = source.convert("RGB")
        source_hash = hashlib.sha256(source_rgb.tobytes()).hexdigest()
    output_root = dataset.paths.object_dir(item.object_id) / "roughness"
    existing = _reusable_marigold_run(
        output_root,
        source_hash=source_hash,
        model_id=analyzer.model_id,
        processing_resolution=analyzer.processing_resolution,
        ensemble_size=MARIGOLD_ENSEMBLE_SIZE,
        seed=cfg.seed,
    )
    if existing is None:
        existing = run_marigold(
            analyzer,
            source_rgb,
            output_root,
            background_remover=background_remover,
            source_label=item.name,
            dataset_id=dataset.dataset_id,
            object_id=item.object_id,
            source_path=item.image.path,
            num_inference_steps=MARIGOLD_INFERENCE_STEPS,
            ensemble_size=MARIGOLD_ENSEMBLE_SIZE,
            seed=cfg.seed,
            crop_padding_ratio=MARIGOLD_CROP_PADDING_RATIO,
            contact_band_fraction=MARIGOLD_CONTACT_BAND_FRACTION,
            mask_erosion_ratio=MARIGOLD_MASK_EROSION_RATIO,
        )
    roughness = existing["roughness"]
    uncertainty = existing.get("roughness_uncertainty") or {}
    quality = existing.get("quality") or {}
    metadata_path = Path(existing["run_dir"]) / "metadata.json"
    item.roughness = RoughnessArtifact(
        metadata_path=str(metadata_path.relative_to(cfg.root)),
        source_image_sha256=source_hash,
        model=str(existing["model"]["id"]),
        mean=float(roughness["mean"]),
        median=float(roughness["median"]),
        std=float(roughness["std"]),
        p25=float(roughness["p25"]),
        p75=float(roughness["p75"]),
        uncertainty_mean=(
            float(uncertainty["mean"]) if uncertainty.get("mean") is not None else None
        ),
        quality_status=str(quality.get("status", "unknown")),
        quality_warnings=[str(item) for item in quality.get("warnings", [])],
        updated_at=str(existing["created_at"]),
    )


def _reusable_marigold_run(
    output_root: Path,
    *,
    source_hash: str,
    model_id: str,
    processing_resolution: int,
    ensemble_size: int,
    seed: int,
) -> dict[str, Any] | None:
    from ..models.marigold_rough import list_saved_runs

    for run in list_saved_runs(output_root):
        source = run.get("source", {})
        model = run.get("model", {})
        if (
            source.get("image_sha256") == source_hash
            and int(run.get("schema_version", 0)) >= 3
            and model.get("id") == model_id
            and model.get("processing_resolution") == processing_resolution
            and model.get("num_inference_steps") == MARIGOLD_INFERENCE_STEPS
            and model.get("ensemble_size") == ensemble_size
            and model.get("seed") == seed
            and run.get("crop", {}).get("padding_ratio") == MARIGOLD_CROP_PADDING_RATIO
            and run.get("scoring", {}).get("contact_band_fraction")
            == MARIGOLD_CONTACT_BAND_FRACTION
            and run.get("scoring", {}).get("mask_erosion_ratio") == MARIGOLD_MASK_EROSION_RATIO
        ):
            return run
    return None


def _prepare_surface_area(
    cfg: Config,
    dataset: Dataset,
    item: DatasetObject,
    *,
    params: Any,
    session: Any,
) -> Any:
    from ..contact_model import analyze_image

    if item.image_2 is None:
        raise FileNotFoundError(f"no image_2 is indexed for {item.name!r}")
    image_path = cfg.root / item.image_2.path
    if not image_path.is_file():
        raise FileNotFoundError(f"image_2 is unavailable for {item.name!r}: {image_path}")
    source_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    run_dir = dataset.paths.object_dir(item.object_id) / "contact_fraction"
    summary_path = run_dir / "summary.json"
    summary = _reusable_surface_summary(
        summary_path,
        source_hash=source_hash,
        image_path=item.image_2.path,
        params=params,
    )
    if summary is None:
        if session is None:
            from ..contact_model import create_rembg_session

            session = create_rembg_session()
        _, summary, _ = analyze_image(
            image_path,
            run_dir,
            item.name,
            params,
            session=session,
        )
        summary["source"] = {
            "view": "image_2",
            "path": item.image_2.path,
            "image_sha256": source_hash,
        }
        write_json_atomic(summary_path, summary)
    _apply_contact_summary(cfg, item, summary, summary_path)
    return session


def _reusable_surface_summary(
    summary_path: Path,
    *,
    source_hash: str,
    image_path: str,
    params: Any,
) -> dict[str, Any] | None:
    if not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        source = summary["source"]
        stored_params = summary["params"]
    except (KeyError, OSError, TypeError, json.JSONDecodeError):
        return None
    expected = {
        "pad_length_mm": params.pad_length_mm,
        "side_angle_deg": params.side_angle_deg,
        "minimum_contact_fraction": params.minimum_contact_fraction,
        "closing_axis": params.closing_axis,
    }
    if params.mode == "rigid":
        expected["rigid_contact_tolerance_mm"] = params.rigid_contact_tolerance_mm
        expected["finger_extension_mm"] = params.finger_extension_mm
        expected["planar_threshold"] = params.planar_threshold
    else:
        expected["minimum_bend_radius_mm"] = params.minimum_bend_radius_mm
    # Summaries written before the rigid mode existed have no "mode" key; they
    # are compliant by construction, so treat a missing key as such.
    stored_mode = stored_params.get("mode", "compliant")
    if (
        source.get("view") == "image_2"
        and source.get("path") == image_path
        and source.get("image_sha256") == source_hash
        and float(summary.get("px_per_mm", -1)) == params.px_per_mm
        and stored_mode == params.mode
        and all(stored_params.get(key) == value for key, value in expected.items())
    ):
        return summary
    return None


def _apply_contact_summary(
    cfg: Config,
    item: DatasetObject,
    summary: dict[str, Any],
    summary_path: Path,
) -> None:
    results = summary["results"]
    item.contact_fraction = ContactFractionArtifact(
        summary_path=str(summary_path.relative_to(cfg.root)),
        schema_version=int(summary["schema_version"]),
        object_height_mm=float(results["object_height_mm"]),
        object_width_mm=float(results["object_width_mm"]),
        geometric_contact_fraction=float(results["geometric_contact_fraction"]),
        combined_contact_fraction=float(results["combined_contact_fraction"]),
        grasp_feasible=bool(results["grasp_feasible"]),
        antipodal_grasp=bool(results["antipodal_grasp"]),
        contact_floor_applied=bool(results["contact_floor_applied"]),
    )
    item.projected_contact_fraction = item.contact_fraction.combined_contact_fraction


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


def _stage_total(dataset: Dataset, stage: PreparationStage) -> int:
    if stage is PreparationStage.EXPERIENCES:
        return len(dataset.objects)
    return len(_surface_items(dataset))


def _task_count(dataset: Dataset, stages: set[PreparationStage]) -> int:
    surface_count = len(_surface_items(dataset))
    return sum(
        1
        if stage in {PreparationStage.INDEX, PreparationStage.EXPERIENCES}
        else surface_count
        for stage in stages
    )


def _surface_items(dataset: Dataset) -> list[DatasetObject]:
    """Return one canonical (preferably baseline) item per physical surface."""
    by_surface: dict[str, DatasetObject] = {}
    for item in dataset.objects.values():
        surface_id = item.surface_id or item.object_id
        existing = by_surface.get(surface_id)
        if existing is None or (
            item.condition_id == "baseline" and existing.condition_id != "baseline"
        ):
            by_surface[surface_id] = item
    return list(by_surface.values())


def _legacy_manifest_view(dataset: Dataset, manifest: PreparationManifest) -> dict:
    descriptions = manifest.stages.get(PreparationStage.DESCRIPTIONS.value, StageStatus())
    embeddings = manifest.stages.get(PreparationStage.EMBEDDINGS.value, StageStatus())
    roughness = manifest.stages.get(PreparationStage.ROUGHNESS.value, StageStatus())
    surface_area = manifest.stages.get(PreparationStage.SURFACE_AREA.value, StageStatus())
    experiences = manifest.stages.get(PreparationStage.EXPERIENCES.value, StageStatus())
    failed = next((item for item in manifest.stages.values() if item.status == "failed"), None)
    return {
        **manifest.model_dump(mode="json"),
        "status": "failed" if failed else "complete",
        "objects": len(dataset.objects),
        "experience_rows": sum(len(item.gripper_outcomes) for item in dataset.objects.values()),
        "descriptors_completed": descriptions.completed,
        "embeddings_completed": embeddings.completed,
        "roughness_completed": roughness.completed,
        "surface_area_completed": surface_area.completed,
        "records_completed": experiences.completed,
        "failed_object": failed.failed_object if failed else None,
        "error": failed.error if failed else None,
    }
