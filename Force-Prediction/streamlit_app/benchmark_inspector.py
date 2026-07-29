"""Reconstruct and render one immutable benchmark prediction for inspection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import streamlit as st

from modules.benchmarking import BenchmarkEvaluation, BenchmarkPredictionBatch
from modules.config import Config
from modules.contracts import ExperienceRecord, Gripper, ObjectRecord
from modules.expforce import pipeline_result_from_dict
from modules.pipeline import PipelineRunResult
from streamlit_app.prediction_ui import render_prediction


@dataclass(frozen=True)
class BenchmarkImageState:
    """Availability and integrity of the image referenced by a saved query."""

    path: Path
    status: Literal["verified", "changed", "unverified", "missing", "unreadable"]
    expected_sha256: str | None
    actual_sha256: str | None

    @property
    def displayable(self) -> bool:
        return self.status in {"verified", "changed", "unverified"}


@dataclass(frozen=True)
class BenchmarkObjectInspection:
    """Fully reconstructed state needed by the shared prediction renderer."""

    row: dict
    detailed: PipelineRunResult
    config: Config
    evaluation_row: dict | None
    truth: ObjectRecord | None
    image: BenchmarkImageState


def benchmark_object_options(rows: list[dict]) -> dict[str, dict]:
    """Return stable, human-readable dropdown options for saved batch rows."""
    options: dict[str, dict] = {}
    for row in rows:
        object_id = str(row["object_id"])
        object_name = str(row.get("object_name") or object_id.replace("_", " ").title())
        options[f"{object_name} · {object_id}"] = row
    return options


def benchmark_config(cfg: Config, batch: BenchmarkPredictionBatch) -> Config:
    """Recreate the display-relevant configuration saved with a batch."""
    metadata = batch.metadata
    scoped = cfg.model_copy(deep=True)
    scoped.prediction.active_grippers = tuple(
        Gripper(name) for name in metadata["active_grippers"]
    )
    scoped.retrieval = type(scoped.retrieval).model_validate(metadata["retrieval"])
    scoped.inputs = type(scoped.inputs).model_validate(metadata["inputs"])
    return scoped


def evaluation_row_for_object(
    evaluation: BenchmarkEvaluation | None,
    object_id: str,
) -> dict | None:
    """Find the selected evaluation version's row without consulting current data."""
    if evaluation is None:
        return None
    matches = [row for row in evaluation.rows if row.get("object_id") == object_id]
    if len(matches) > 1:
        raise ValueError(f"evaluation contains duplicate rows for {object_id!r}")
    return matches[0] if matches else None


def truth_from_evaluation_row(
    row: dict | None,
    active_grippers: tuple[Gripper, ...],
    *,
    force_limit_n: float,
) -> ObjectRecord | None:
    """Rebuild the exact saved truth needed by ``render_prediction``."""
    if row is None:
        return None

    records: dict[str, ExperienceRecord] = {}
    for gripper in active_grippers:
        name = gripper.value
        feasible = row.get(f"true_{name}_feasible")
        force = row.get(f"true_{name}_force_n")
        if feasible is None:
            raise ValueError(
                f"evaluation row for {row.get('object_id')!r} is missing {name} truth"
            )
        records[name] = ExperienceRecord(
            object_id=str(row["object_id"]),
            surface_id=row.get("surface_id"),
            condition_id=str(row.get("condition_id") or "baseline"),
            image_path=str(row.get("image_path") or ""),
            mass_g=row.get("mass_g"),
            roughness_index=row.get("roughness_index"),
            projected_contact_fraction=row.get("projected_contact_fraction"),
            gripper=gripper,
            min_force_n=force if feasible else None,
            feasible=bool(feasible),
            failed_at_limit_n=None if feasible else force_limit_n,
            semantic_description=str(row.get("semantic_description") or ""),
        )

    return ObjectRecord(
        object_id=str(row["object_id"]),
        surface_id=row.get("surface_id"),
        condition_id=str(row.get("condition_id") or "baseline"),
        gecko=records.get("gecko"),
        silicone=records.get("silicone"),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark_image_state(cfg: Config, row: dict) -> BenchmarkImageState:
    """Resolve the saved query path and verify it against the batch-time hash."""
    raw_path = str(row.get("image_path") or "")
    path = cfg.root / raw_path
    expected = row.get("image_sha256")
    if not path.is_file():
        return BenchmarkImageState(path, "missing", expected, None)
    try:
        actual = _file_sha256(path)
    except OSError:
        return BenchmarkImageState(path, "unreadable", expected, None)
    if expected is None:
        status: Literal["verified", "changed", "unverified"] = "unverified"
    else:
        status = "verified" if actual == expected else "changed"
    return BenchmarkImageState(path, status, expected, actual)


def build_benchmark_inspection(
    cfg: Config,
    batch: BenchmarkPredictionBatch,
    row: dict,
    evaluation: BenchmarkEvaluation | None,
) -> BenchmarkObjectInspection:
    """Reconstruct one saved prediction and its selected historical evaluation."""
    scoped = benchmark_config(cfg, batch)
    evaluation_row = evaluation_row_for_object(evaluation, str(row["object_id"]))
    truth = truth_from_evaluation_row(
        evaluation_row,
        scoped.prediction.active_grippers,
        force_limit_n=scoped.force.limit_n,
    )
    return BenchmarkObjectInspection(
        row=row,
        detailed=pipeline_result_from_dict(row["pipeline_result"]),
        config=scoped,
        evaluation_row=evaluation_row,
        truth=truth,
        image=benchmark_image_state(scoped, row),
    )


def _format_backend(backend: object) -> str:
    if isinstance(backend, dict):
        return " · ".join(f"{name}: {value}" for name, value in backend.items())
    return str(backend or "Unknown")


def _render_query_panel(
    inspection: BenchmarkObjectInspection,
    batch: BenchmarkPredictionBatch,
) -> None:
    row = inspection.row
    metadata = batch.metadata
    st.subheader("Exact benchmark query")
    st.write(f"**{row.get('object_name') or row['object_id']}**")
    st.caption(f"Object ID: `{row['object_id']}`")

    if inspection.image.displayable:
        st.image(str(inspection.image.path), width="stretch")
    if inspection.image.status == "changed":
        st.warning(
            "The file at the saved image path has changed since this benchmark ran. "
            "The current file is displayed, but it is not the exact benchmark image."
        )
    elif inspection.image.status == "missing":
        st.warning("The image referenced by this benchmark is no longer available.")
    elif inspection.image.status == "unreadable":
        st.warning("The image referenced by this benchmark could not be read.")
    elif inspection.image.status == "unverified":
        st.caption("This older row did not save an image hash, so integrity is unverified.")
    else:
        st.caption(f"Verified image SHA-256: `{inspection.image.actual_sha256}`")

    st.metric(
        "Mass",
        f"{row['mass_g']:.1f} g" if row.get("mass_g") is not None else "Not recorded",
    )
    sensors = st.columns(2)
    sensors[0].metric(
        "Roughness index",
        f"{row['roughness_index']:g}"
        if row.get("roughness_index") is not None
        else "Not recorded",
    )
    sensors[1].metric(
        "Contact",
        f"{row['projected_contact_fraction']:.3f}"
        if row.get("projected_contact_fraction") is not None
        else "Not recorded",
    )

    test_ids = set(metadata.get("test_ids", ()))
    split_label = "Test" if row["object_id"] in test_ids else "Benchmark query"
    st.write(f"**Dataset split:** {split_label}")
    st.write(f"**Surface/condition:** {row.get('surface_id')} / {row.get('condition_id')}")
    st.write(f"**Description:** {row.get('semantic_description') or 'Not recorded'}")

    st.subheader("Run configuration")
    st.write(
        f"**Experiment:** {metadata['experiment'].upper()} · "
        f"{metadata['experiment_method']}"
    )
    st.write(f"**Created:** {metadata['created_at']}")
    st.write(
        f"**Protocol:** {str(metadata.get('evaluation_protocol', 'unknown')).replace('_', ' ')}"
    )
    st.write(f"**Active grippers:** {', '.join(metadata['active_grippers'])}")
    st.write(f"**Backend:** {_format_backend(metadata.get('backend'))}")
    st.write(f"**VLM:** {metadata.get('model', 'Unknown')}")
    st.write(f"**Text embedding:** {metadata.get('embedding_model', 'Unknown')}")
    with st.expander("Saved retrieval and input configuration"):
        st.json(
            {
                "retrieval": metadata["retrieval"],
                "inputs": metadata["inputs"],
                "reference_ids": metadata.get("reference_ids", []),
            },
            expanded=True,
        )
    with st.expander("Experiment definition"):
        st.json(metadata.get("experiment_definition", {}), expanded=True)


def render_benchmark_object_inspector(
    context_cfg: Config,
    batch: BenchmarkPredictionBatch,
    evaluation: BenchmarkEvaluation | None,
) -> None:
    """Render the Single Run-style detail view for one row in a saved batch."""
    options = benchmark_object_options(batch.rows)
    if not options:
        st.info("This prediction batch contains no object rows.")
        return
    selected = st.selectbox(
        "Benchmark object",
        list(options),
        key=f"benchmark_object_inspector_{batch.batch_id}",
    )
    try:
        inspection = build_benchmark_inspection(
            context_cfg,
            batch,
            options[selected],
            evaluation,
        )
    except (KeyError, TypeError, ValueError) as exc:
        st.error(f"This saved benchmark row could not be reconstructed: {exc}")
        return

    query_col, output_col = st.columns([0.34, 0.66], gap="large")
    with query_col:
        _render_query_panel(inspection, batch)

    with output_col:
        st.subheader("Pipeline output")
        if evaluation is None:
            unscored_message = "Not evaluated: this batch has no saved evaluation version."
        elif inspection.evaluation_row is None:
            unscored_message = (
                "Not evaluated in the selected evaluation version because complete truth "
                "was unavailable."
            )
        else:
            unscored_message = None
            st.caption(f"Evaluation version: `{evaluation.evaluation_id}`")
        render_prediction(
            inspection.detailed,
            inspection.truth,
            counterfactual=False,
            baseline=None,
            cfg=inspection.config,
            experiment=batch.metadata["experiment"],
            unscored_message=unscored_message,
            truth_context_label="Selected benchmark evaluation truth",
        )
        with st.expander("Raw saved prediction payload"):
            st.json(inspection.row["pipeline_result"], expanded=True)
