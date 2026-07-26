"""Dataset-scoped artifact loading and atomic persistence."""

from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from ..contracts import ExperienceRecord, Meta, load_experiences, save_experiences
from .models import (
    Dataset,
    DescriptionArtifact,
    EmbeddingArtifact,
    PreparationManifest,
    PreparedObjectCheckpoint,
)


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
        temporary = Path(fh.name)
    temporary.replace(path)


def checkpoint_path(dataset: Dataset, object_id: str) -> Path:
    return dataset.paths.descriptors / f"{object_id}.json"


def load_checkpoints(dataset: Dataset) -> dict[str, PreparedObjectCheckpoint]:
    if not dataset.paths.descriptors.exists():
        return {}
    output: dict[str, PreparedObjectCheckpoint] = {}
    for path in sorted(dataset.paths.descriptors.glob("*.json")):
        try:
            item = PreparedObjectCheckpoint.model_validate_json(path.read_text())
        except (OSError, ValueError):
            continue
        if item.dataset_id == "expforce" and dataset.dataset_id != "expforce":
            item.dataset_id = dataset.dataset_id
        output[item.object_id] = item
    return output


def attach_checkpoints(dataset: Dataset) -> Dataset:
    checkpoints = load_checkpoints(dataset)
    for object_id, item in dataset.objects.items():
        checkpoint = checkpoints.get(object_id)
        if checkpoint is None:
            continue
        item.description = DescriptionArtifact(
            value=checkpoint.descriptor,
            source=checkpoint.descriptor_source,
            model=checkpoint.descriptor_model,
            signature=checkpoint.descriptor_signature,
            image_sha256=checkpoint.image_sha256,
            updated_at=checkpoint.updated_at,
        )
        item.embedding = EmbeddingArtifact(
            status=checkpoint.embedding_status,
            model=checkpoint.embedding_model,
            dim=checkpoint.embedding_dim,
            descriptor_sha256=checkpoint.embedding_descriptor_sha256,
            cache_key=checkpoint.embedding_cache_key,
            vector_sha256=checkpoint.embedding_sha256,
        )
    dataset.capabilities.has_descriptions = bool(dataset.descriptions)
    dataset.capabilities.has_embeddings = any(
        item.status == "ready" for item in dataset.embeddings.values()
    )
    return dataset


def load_manifest(dataset: Dataset) -> PreparationManifest:
    path = dataset.paths.preparation_manifest
    if path.exists():
        try:
            return PreparationManifest.model_validate_json(path.read_text())
        except (OSError, ValueError):
            pass
    return PreparationManifest(
        dataset_id=dataset.dataset_id,
        source_fingerprint=dataset.source_fingerprint,
    )


def save_manifest(dataset: Dataset, manifest: PreparationManifest) -> None:
    manifest.updated_at = datetime.now(UTC).isoformat()
    write_json_atomic(
        dataset.paths.preparation_manifest,
        manifest.model_dump(mode="json"),
    )


def load_dataset_experiences(dataset: Dataset) -> list[ExperienceRecord]:
    return load_experiences(dataset.paths.experiences)


def build_dataset_experiences(dataset: Dataset, force_limit_n: float) -> list[ExperienceRecord]:
    if not dataset.capabilities.can_build_experiences:
        raise ValueError(
            f"dataset {dataset.display_name!r} lacks complete measurements and paired labels"
        )
    records: list[ExperienceRecord] = []
    for item in dataset.objects.values():
        description = item.description.value.description if item.description else item.name
        assert item.mass_g is not None
        assert item.roughness_class is not None
        assert item.projected_contact_fraction is not None
        for gripper, outcome in item.gripper_outcomes.items():
            records.append(
                ExperienceRecord(
                    object_id=item.object_id,
                    image_path=item.image.path,
                    mass_g=item.mass_g,
                    roughness_class=item.roughness_class,
                    projected_contact_fraction=item.projected_contact_fraction,
                    gripper=gripper,
                    min_force_n=outcome.min_force_n if outcome.feasible else None,
                    feasible=outcome.feasible,
                    failed_at_limit_n=None if outcome.feasible else force_limit_n,
                    semantic_description=description,
                    meta=Meta(pad_id=f"{dataset.dataset_id}-dataset"),
                )
            )
    save_experiences(dataset.paths.experiences, records)
    return records


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
