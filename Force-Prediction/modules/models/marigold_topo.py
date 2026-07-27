"""Marigold normal-based topographic analysis and artifact persistence.

Appearance roughness describes a BRDF. This module instead estimates local angular
variation after removing the object's smooth base curvature, producing a separate
topographic-roughness proxy for bumps, ridges, and grooves.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .background_remover import BackgroundRemover
from .marigold_rough import (
    DEFAULT_ALPHA_THRESHOLD,
    DEFAULT_CONTACT_BAND_FRACTION,
    DEFAULT_CROP_PADDING_RATIO,
    DEFAULT_MASK_EROSION_RATIO,
    MIN_MASK_BBOX_FILL_FRACTION,
    _bbox_fill_fraction,
    _central_principal_axis_band,
    _erode_mask,
    _foreground_preview,
    _image_sha256,
    _padded_foreground_bbox,
    available_device,
)

DEFAULT_NORMALS_MODEL_ID = "prs-eth/marigold-normals-v1-1"
DEFAULT_PROCESSING_RESOLUTION = 768
DEFAULT_INFERENCE_STEPS = 4
DEFAULT_ENSEMBLE_SIZE = 3
DEFAULT_BASE_SURFACE_SIGMA_RATIO = 0.04
DEFAULT_DISPLAY_MAX_ANGLE_DEG = 20.0
SCHEMA_VERSION = 1
MIN_SCORING_PIXELS = 256


@dataclass(frozen=True)
class NormalPrediction:
    """Unit camera-space normals and optional angular ensemble uncertainty."""

    normals: np.ndarray
    processed_size: tuple[int, int]
    uncertainty: np.ndarray | None = None


class MarigoldNormalsAnalyzer:
    """Lazy adapter for the Marigold v1.1 monocular normals checkpoint."""

    def __init__(
        self,
        *,
        device: str | None = None,
        processing_resolution: int = DEFAULT_PROCESSING_RESOLUTION,
        model_id: str = DEFAULT_NORMALS_MODEL_ID,
    ) -> None:
        if processing_resolution < 64:
            raise ValueError("processing_resolution must be at least 64 pixels")
        self.device = device or available_device()
        self.processing_resolution = processing_resolution
        self.model_id = model_id
        self.pipe: Any | None = None

    def _load(self) -> None:
        try:
            import torch
            from diffusers import MarigoldNormalsPipeline
        except ImportError as error:
            raise RuntimeError(
                "Marigold normals requires the optional roughness dependencies. Install "
                "this project with `pip install -e '.[roughness]'`."
            ) from error

        dtype = torch.float32 if self.device == "cpu" else torch.float16
        kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if dtype == torch.float16:
            kwargs["variant"] = "fp16"
        try:
            self.pipe = MarigoldNormalsPipeline.from_pretrained(self.model_id, **kwargs)
        except Exception:
            kwargs.pop("variant", None)
            self.pipe = MarigoldNormalsPipeline.from_pretrained(self.model_id, **kwargs)
        self.pipe = self.pipe.to(self.device)
        self.pipe.enable_attention_slicing()
        self.pipe.vae.enable_tiling()

    def analyze(
        self,
        image: Image.Image,
        *,
        num_inference_steps: int = DEFAULT_INFERENCE_STEPS,
        ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
        seed: int = 2024,
    ) -> NormalPrediction:
        """Estimate an ensembled unit-normal map at the configured resolution."""
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        if ensemble_size < 1:
            raise ValueError("ensemble_size must be positive")
        if self.pipe is None:
            self._load()
        assert self.pipe is not None

        import torch

        prepared = image.convert("RGB")
        if self.device == "mps" and ensemble_size > 1:
            members = []
            for member in range(ensemble_size):
                generator = torch.Generator(device=self.device).manual_seed(seed + member)
                result = self.pipe(
                    prepared,
                    num_inference_steps=num_inference_steps,
                    ensemble_size=1,
                    processing_resolution=self.processing_resolution,
                    match_input_resolution=False,
                    output_uncertainty=False,
                    generator=generator,
                )
                members.append(np.asarray(result.prediction[0], dtype=np.float32))
            stacked = np.stack(members, axis=0)
            normals = _normalize_vectors(stacked.mean(axis=0))
            similarities = np.clip(np.sum(stacked * normals[None, ...], axis=-1), -1, 1)
            uncertainty = (
                np.mean(np.arccos(similarities), axis=0) / np.pi if ensemble_size > 2 else None
            )
        else:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            result = self.pipe(
                prepared,
                num_inference_steps=num_inference_steps,
                ensemble_size=ensemble_size,
                processing_resolution=self.processing_resolution,
                match_input_resolution=False,
                output_uncertainty=ensemble_size > 2,
                ensembling_kwargs={"reduction": "mean"},
                generator=generator,
            )
            normals = np.asarray(result.prediction[0], dtype=np.float32)
            uncertainty = (
                np.squeeze(np.asarray(result.uncertainty[0], dtype=np.float32))
                if result.uncertainty is not None
                else None
            )
        if self.device == "mps":
            torch.mps.empty_cache()
        return NormalPrediction(
            normals=_normalize_vectors(normals),
            processed_size=(normals.shape[1], normals.shape[0]),
            uncertainty=uncertainty,
        )


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-8)


def _masked_smooth_normals(
    normals: np.ndarray,
    foreground: np.ndarray,
    sigma_pixels: float,
) -> np.ndarray:
    """Gaussian-smooth vectors without bleeding the background into the object."""
    weights = cv2.GaussianBlur(
        foreground.astype(np.float32),
        (0, 0),
        sigmaX=sigma_pixels,
        sigmaY=sigma_pixels,
        borderType=cv2.BORDER_REFLECT,
    )
    smoothed = np.empty_like(normals, dtype=np.float32)
    for channel in range(3):
        weighted = cv2.GaussianBlur(
            normals[..., channel] * foreground,
            (0, 0),
            sigmaX=sigma_pixels,
            sigmaY=sigma_pixels,
            borderType=cv2.BORDER_REFLECT,
        )
        smoothed[..., channel] = weighted / np.maximum(weights, 1e-6)
    return _normalize_vectors(smoothed)


def _object_short_span(mask: np.ndarray) -> int:
    ys, xs = np.where(mask)
    if not xs.size:
        raise RuntimeError("The saved foreground mask contains no foreground pixels.")
    return min(int(xs.max()) - int(xs.min()) + 1, int(ys.max()) - int(ys.min()) + 1)


def _statistics(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise RuntimeError("No finite topographic values remain in the scoring region.")
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "p25": float(np.percentile(finite, 25)),
        "p75": float(np.percentile(finite, 75)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
    }


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "image"


def _resize_mask(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    resized = image.resize(size, Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8) >= 128


def _normal_visualization(normals: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    display = np.clip((normals + 1) * 127.5, 0, 255).astype(np.uint8)
    return display * foreground[..., None]


def _run_prepared_topography(
    analyzer: MarigoldNormalsAnalyzer,
    crop: Image.Image,
    foreground_source: Image.Image,
    scoring_source: Image.Image,
    output_root: Path,
    *,
    source: dict[str, Any],
    num_inference_steps: int,
    ensemble_size: int,
    seed: int,
    base_surface_sigma_ratio: float,
    display_max_angle_deg: float,
    run_key: str | None,
    started_at: datetime,
    initial_warnings: list[str] | None = None,
    extra_artifacts: dict[str, tuple[Image.Image, str]] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prediction = analyzer.analyze(
        crop,
        num_inference_steps=num_inference_steps,
        ensemble_size=ensemble_size,
        seed=seed,
    )
    foreground = _resize_mask(foreground_source, prediction.processed_size)
    scoring_mask = _resize_mask(scoring_source, prediction.processed_size) & foreground
    if int(scoring_mask.sum()) < MIN_SCORING_PIXELS:
        raise RuntimeError("The grasp-region mask is too small for topographic analysis.")

    short_span = _object_short_span(foreground)
    sigma_pixels = max(2.0, short_span * base_surface_sigma_ratio)
    base_normals = _masked_smooth_normals(prediction.normals, foreground, sigma_pixels)
    cosine = np.clip(np.sum(prediction.normals * base_normals, axis=-1), -1, 1)
    bump_angle_deg = np.degrees(np.arccos(cosine)).astype(np.float32)
    angle_stats = _statistics(bump_angle_deg[scoring_mask])
    topographic_score = float(np.clip(angle_stats["p75"] / display_max_angle_deg, 0, 1))
    uncertainty_stats = (
        _statistics(prediction.uncertainty[scoring_mask])
        if prediction.uncertainty is not None
        else None
    )

    warnings = list(initial_warnings or [])
    if uncertainty_stats is not None and uncertainty_stats["mean"] > 0.05:
        warnings.append("high_normal_uncertainty")
    if angle_stats["p95"] >= display_max_angle_deg:
        warnings.append("display_angle_clipping")

    created_at = datetime.now(UTC)
    source_label = str(source.get("label") or "image")
    run_id = _slug(run_key) if run_key is not None else (
        f"{created_at:%Y%m%dT%H%M%S%fZ}_{_slug(source_label)}"
    )
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=run_key is not None)

    artifacts: dict[str, str] = {}
    for artifact, (artifact_image, filename) in (extra_artifacts or {}).items():
        artifact_image.save(run_dir / filename)
        artifacts[artifact] = filename
    crop.save(run_dir / "input_crop.png")
    Image.fromarray(foreground.astype(np.uint8) * 255, mode="L").save(
        run_dir / "foreground_mask.png"
    )
    Image.fromarray(scoring_mask.astype(np.uint8) * 255, mode="L").save(
        run_dir / "scoring_mask.png"
    )
    Image.fromarray(_normal_visualization(prediction.normals, foreground)).save(
        run_dir / "normal_map.png"
    )
    Image.fromarray(_normal_visualization(base_normals, foreground)).save(
        run_dir / "base_normal_map.png"
    )
    bump_display = (
        np.clip(bump_angle_deg / display_max_angle_deg, 0, 1)
        * foreground.astype(np.float32)
        * 255
    ).astype(np.uint8)
    Image.fromarray(bump_display, mode="L").save(run_dir / "bump_angle.png")
    artifacts.update(
        {
            "input_crop": "input_crop.png",
            "foreground_mask": "foreground_mask.png",
            "scoring_mask": "scoring_mask.png",
            "normal_map": "normal_map.png",
            "base_normal_map": "base_normal_map.png",
            "bump_angle": "bump_angle.png",
        }
    )
    if prediction.uncertainty is not None:
        uncertainty_display = (
            np.clip(prediction.uncertainty, 0, 1) * foreground.astype(np.float32) * 255
        ).astype(np.uint8)
        Image.fromarray(uncertainty_display, mode="L").save(
            run_dir / "normal_uncertainty.png"
        )
        artifacts["normal_uncertainty"] = "normal_uncertainty.png"

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": created_at.isoformat(),
        "duration_seconds": round((created_at - started_at).total_seconds(), 3),
        "source": source,
        "model": {
            "id": analyzer.model_id,
            "device": analyzer.device,
            "processing_resolution": analyzer.processing_resolution,
            "processed_size": list(prediction.processed_size),
            "num_inference_steps": num_inference_steps,
            "ensemble_size": ensemble_size,
            "seed": seed,
        },
        "method": {
            "name": "local_normal_angular_residual",
            "base_surface_sigma_ratio": base_surface_sigma_ratio,
            "base_surface_sigma_pixels": sigma_pixels,
            "display_max_angle_deg": display_max_angle_deg,
            "scoring_pixels": int(scoring_mask.sum()),
            "interpretation": (
                "Uncalibrated geometric texture proxy from monocular normals; larger values "
                "indicate stronger local bumps, ridges, or grooves."
            ),
        },
        "topographic_roughness": {
            "score_0_1": topographic_score,
            "angle_degrees": angle_stats,
        },
        "normal_uncertainty": uncertainty_stats,
        "quality": {
            "status": "warning" if warnings else "ok",
            "warnings": warnings,
        },
        "artifacts": artifacts,
    }
    metadata.update(extra_metadata or {})
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    metadata["run_dir"] = str(run_dir)
    return metadata


def run_topographic_roughness(
    analyzer: MarigoldNormalsAnalyzer,
    appearance_run_dir: Path,
    output_root: Path,
    *,
    num_inference_steps: int = DEFAULT_INFERENCE_STEPS,
    ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
    seed: int = 2024,
    base_surface_sigma_ratio: float = DEFAULT_BASE_SURFACE_SIGMA_RATIO,
    display_max_angle_deg: float = DEFAULT_DISPLAY_MAX_ANGLE_DEG,
    run_key: str | None = None,
) -> dict[str, Any]:
    """Measure local normal residuals using a saved crop and grasp-region mask.

    ``run_key`` selects a stable output directory. Reusing it updates that run in
    place; omitting it retains timestamped history behavior for batch scripts.
    """
    if not 0 < base_surface_sigma_ratio <= 0.25:
        raise ValueError("base_surface_sigma_ratio must be between 0 and 0.25")
    if display_max_angle_deg <= 0:
        raise ValueError("display_max_angle_deg must be positive")
    metadata_path = appearance_run_dir / "metadata.json"
    appearance = json.loads(metadata_path.read_text())
    crop_path = appearance_run_dir / appearance["artifacts"]["inference_crop"]
    foreground_name = appearance["artifacts"].get(
        "analysis_foreground_mask",
        appearance["artifacts"]["inference_mask"],
    )
    foreground_path = appearance_run_dir / foreground_name
    scoring_path = appearance_run_dir / appearance["artifacts"]["scoring_mask"]
    with Image.open(crop_path) as image:
        crop = image.convert("RGB")
    # Decode the small mask inputs before loading/running the diffusion model. This
    # avoids sporadic image-decoder failures when MPS memory pressure is highest.
    with Image.open(foreground_path) as image:
        foreground_source = image.convert("L").copy()
    with Image.open(scoring_path) as image:
        scoring_source = image.convert("L").copy()

    started_at = datetime.now(UTC)
    return _run_prepared_topography(
        analyzer,
        crop,
        foreground_source,
        scoring_source,
        output_root,
        source={
            "dataset_id": appearance.get("source", {}).get("dataset_id"),
            "object_id": appearance.get("source", {}).get("object_id"),
            "label": str(appearance.get("source", {}).get("label") or "image"),
            "appearance_run_id": appearance.get("run_id"),
            "appearance_metadata": str(metadata_path),
        },
        num_inference_steps=num_inference_steps,
        ensemble_size=ensemble_size,
        seed=seed,
        base_surface_sigma_ratio=base_surface_sigma_ratio,
        display_max_angle_deg=display_max_angle_deg,
        run_key=run_key,
        started_at=started_at,
    )


def run_marigold_topography(
    analyzer: MarigoldNormalsAnalyzer,
    image: Image.Image,
    output_root: Path,
    *,
    background_remover: BackgroundRemover | None = None,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
    source_label: str,
    dataset_id: str | None = None,
    object_id: str | None = None,
    source_path: str | None = None,
    num_inference_steps: int = DEFAULT_INFERENCE_STEPS,
    ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
    seed: int = 2024,
    crop_padding_ratio: float = DEFAULT_CROP_PADDING_RATIO,
    contact_band_fraction: float = DEFAULT_CONTACT_BAND_FRACTION,
    mask_erosion_ratio: float = DEFAULT_MASK_EROSION_RATIO,
    base_surface_sigma_ratio: float = DEFAULT_BASE_SURFACE_SIGMA_RATIO,
    display_max_angle_deg: float = DEFAULT_DISPLAY_MAX_ANGLE_DEG,
    run_key: str | None = None,
) -> dict[str, Any]:
    """Run normal-based topography directly from an RGB image.

    The central major-axis band excludes object ends such as a can's lid while
    treating the remaining visible surface as approximately uniform.
    """
    if not 0 <= alpha_threshold <= 255:
        raise ValueError("alpha_threshold must be between 0 and 255")
    if not 0 <= crop_padding_ratio <= 1:
        raise ValueError("crop_padding_ratio must be between 0 and 1")
    if not 0 < contact_band_fraction <= 1:
        raise ValueError("contact_band_fraction must be between 0 and 1")
    if not 0 <= mask_erosion_ratio <= 0.25:
        raise ValueError("mask_erosion_ratio must be between 0 and 0.25")
    if not 0 < base_surface_sigma_ratio <= 0.25:
        raise ValueError("base_surface_sigma_ratio must be between 0 and 0.25")
    if display_max_angle_deg <= 0:
        raise ValueError("display_max_angle_deg must be positive")

    started_at = datetime.now(UTC)
    input_rgb = image.convert("RGB")
    input_array = np.asarray(input_rgb, dtype=np.uint8)
    if background_remover is not None:
        background = background_remover.remove(input_rgb)
        alpha = background.alpha
        cutout_rgba = background.cutout_rgba
        background_model = background_remover.model_name
    else:
        alpha = np.full(input_array.shape[:2], 255, dtype=np.uint8)
        cutout_rgba = np.dstack((input_array, alpha))
        background_model = None

    source_foreground = alpha >= alpha_threshold
    crop_bbox = _padded_foreground_bbox(source_foreground, crop_padding_ratio)
    x0, y0, x1, y1 = crop_bbox
    crop = input_rgb.crop(crop_bbox)
    crop_alpha = alpha[y0:y1, x0:x1]
    foreground = crop_alpha >= alpha_threshold
    if not foreground.any():
        raise RuntimeError("Background removal found no foreground pixels in this image.")

    warnings: list[str] = []
    bbox_fill_fraction = _bbox_fill_fraction(source_foreground)
    if background_remover is None:
        warnings.append("background_removal_disabled")
    if bbox_fill_fraction < MIN_MASK_BBOX_FILL_FRACTION:
        warnings.append("sparse_foreground_mask")
    eroded_foreground, erosion_radius = _erode_mask(foreground, mask_erosion_ratio)
    scoring_mask = _central_principal_axis_band(eroded_foreground, contact_band_fraction)
    if int(eroded_foreground.sum()) < MIN_SCORING_PIXELS:
        warnings.append("erosion_removed_too_much_foreground")
        eroded_foreground = foreground
        scoring_mask = _central_principal_axis_band(foreground, contact_band_fraction)
        erosion_radius = 0
    if int(scoring_mask.sum()) < MIN_SCORING_PIXELS:
        warnings.append("contact_band_too_small")
        scoring_mask = eroded_foreground

    foreground_source = Image.fromarray(foreground.astype(np.uint8) * 255, mode="L")
    scoring_source = Image.fromarray(scoring_mask.astype(np.uint8) * 255, mode="L")
    return _run_prepared_topography(
        analyzer,
        crop,
        foreground_source,
        scoring_source,
        output_root,
        source={
            "dataset_id": dataset_id,
            "object_id": object_id,
            "label": source_label,
            "path": source_path,
            "image_sha256": _image_sha256(input_rgb),
            "original_size": list(input_rgb.size),
        },
        num_inference_steps=num_inference_steps,
        ensemble_size=ensemble_size,
        seed=seed,
        base_surface_sigma_ratio=base_surface_sigma_ratio,
        display_max_angle_deg=display_max_angle_deg,
        run_key=run_key,
        started_at=started_at,
        initial_warnings=warnings,
        extra_artifacts={
            "input": (input_rgb, "input.png"),
            "foreground_preview": (
                Image.fromarray(_foreground_preview(input_array, alpha)),
                "foreground_preview.png",
            ),
            "cutout": (Image.fromarray(cutout_rgba, mode="RGBA"), "cutout.png"),
            "source_foreground_mask": (
                Image.fromarray(alpha, mode="L"),
                "source_foreground_mask.png",
            ),
        },
        extra_metadata={
            "background_removal": {
                "enabled": background_remover is not None,
                "model": background_model,
                "alpha_threshold": alpha_threshold,
                "foreground_fraction": float(np.mean(source_foreground)),
                "bbox_fill_fraction": bbox_fill_fraction,
            },
            "crop": {
                "bbox_xyxy": list(crop_bbox),
                "padding_ratio": crop_padding_ratio,
                "size": list(crop.size),
            },
            "scoring": {
                "strategy": "eroded_central_principal_axis_band",
                "contact_band_fraction": contact_band_fraction,
                "mask_erosion_ratio": mask_erosion_ratio,
                "erosion_radius_pixels": erosion_radius,
            },
        },
    )


def list_saved_topography_runs(output_root: Path) -> list[dict[str, Any]]:
    """Load valid topographic-run metadata, newest first."""
    runs: list[dict[str, Any]] = []
    if not output_root.is_dir():
        return runs
    for metadata_path in output_root.glob("*/metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(metadata, dict):
            metadata["run_dir"] = str(metadata_path.parent)
            runs.append(metadata)
    return sorted(runs, key=lambda item: str(item.get("created_at", "")), reverse=True)
