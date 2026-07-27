"""Marigold IID appearance-roughness inference and artifact persistence.

Heavy optional dependencies are imported only when inference is requested so the
rest of the force-prediction application remains usable without Marigold installed.
"""

from __future__ import annotations

import hashlib
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

DEFAULT_MODEL_ID = "prs-eth/marigold-iid-appearance-v1-1"
SCHEMA_VERSION = 3
DEFAULT_ALPHA_THRESHOLD = 120
DEFAULT_PROCESSING_RESOLUTION = 768
DEFAULT_INFERENCE_STEPS = 4
DEFAULT_ENSEMBLE_SIZE = 3
DEFAULT_CROP_PADDING_RATIO = 0.15
DEFAULT_CONTACT_BAND_FRACTION = 0.60
DEFAULT_MASK_EROSION_RATIO = 0.01
MIN_MASK_BBOX_FILL_FRACTION = 0.50
MIN_SCORING_PIXELS = 256


@dataclass(frozen=True)
class IntrinsicPrediction:
    """Marigold appearance outputs at the model processing resolution."""

    albedo_rgb: np.ndarray
    roughness: np.ndarray
    metallicity: np.ndarray
    processed_size: tuple[int, int]
    roughness_uncertainty: np.ndarray | None = None
    metallicity_uncertainty: np.ndarray | None = None


def available_device() -> str:
    """Return the best available Torch device without importing Torch at module load."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Marigold requires the optional roughness dependencies. Install this project "
            "with `pip install -e '.[roughness]'`."
        ) from error
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class MarigoldAnalyzer:
    """Lazy Marigold intrinsic decomposition model.

    The pipeline is retained on the instance after its first run. Streamlit caches
    this object as a resource, avoiding a multi-gigabyte reload on every rerun.
    """

    def __init__(
        self,
        *,
        device: str | None = None,
        processing_resolution: int = DEFAULT_PROCESSING_RESOLUTION,
        model_id: str = DEFAULT_MODEL_ID,
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
            from diffusers import MarigoldIntrinsicsPipeline
        except ImportError as error:
            raise RuntimeError(
                "Marigold requires the optional roughness dependencies. Install this project "
                "with `pip install -e '.[roughness]'`."
            ) from error

        dtype = torch.float32 if self.device == "cpu" else torch.float16
        kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if dtype == torch.float16:
            kwargs["variant"] = "fp16"
        try:
            self.pipe = MarigoldIntrinsicsPipeline.from_pretrained(self.model_id, **kwargs)
        except Exception:
            # Some model revisions do not publish a separate fp16 variant.
            kwargs.pop("variant", None)
            self.pipe = MarigoldIntrinsicsPipeline.from_pretrained(self.model_id, **kwargs)
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
    ) -> IntrinsicPrediction:
        """Decompose an RGB image into albedo, roughness, and metallicity."""
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        if ensemble_size < 1:
            raise ValueError("ensemble_size must be positive")
        if self.pipe is None:
            self._load()
        assert self.pipe is not None

        import torch

        prepared = image.convert("RGB")
        uncertainty: np.ndarray | None
        if self.device == "mps" and ensemble_size > 1:
            # Diffusers' median ensemble sorts a 5-D tensor. Apple's MPS sort kernel
            # supports only four axes, so combine independent predictions on CPU.
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
                members.append(np.asarray(result.prediction, dtype=np.float32))
            stacked = np.stack(members, axis=0)
            prediction = np.median(stacked, axis=0)
            uncertainty = (
                np.median(np.abs(stacked - prediction[None, ...]), axis=0)
                if ensemble_size > 2
                else None
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
                generator=generator,
            )
            prediction = np.asarray(result.prediction)
            uncertainty = np.asarray(result.uncertainty) if result.uncertainty is not None else None
        properties = self.pipe.target_properties
        material_index = properties["target_names"].index("material")
        material_names = properties["material"]["sub_target_names"]
        material = prediction[material_index]
        roughness = np.squeeze(material[..., material_names.index("roughness")]).astype(np.float32)
        metallicity = np.squeeze(material[..., material_names.index("metallicity")]).astype(
            np.float32
        )
        roughness_uncertainty = None
        metallicity_uncertainty = None
        if uncertainty is not None:
            material_uncertainty = uncertainty[material_index]
            roughness_uncertainty = np.squeeze(
                material_uncertainty[..., material_names.index("roughness")]
            ).astype(np.float32)
            metallicity_uncertainty = np.squeeze(
                material_uncertainty[..., material_names.index("metallicity")]
            ).astype(np.float32)

        visualized = self.pipe.image_processor.visualize_intrinsics(prediction, properties)
        albedo = np.asarray(visualized[0]["albedo"].convert("RGB"), dtype=np.uint8)
        if self.device == "mps":
            torch.mps.empty_cache()
        return IntrinsicPrediction(
            albedo_rgb=albedo,
            roughness=roughness,
            metallicity=metallicity,
            processed_size=(roughness.shape[1], roughness.shape[0]),
            roughness_uncertainty=roughness_uncertainty,
            metallicity_uncertainty=metallicity_uncertainty,
        )


def _statistics(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            key: float("nan")
            for key in ("mean", "median", "std", "min", "max", "p05", "p25", "p75", "p95")
        }
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p25": float(np.percentile(finite, 25)),
        "p75": float(np.percentile(finite, 75)),
        "p95": float(np.percentile(finite, 95)),
    }


def _padded_foreground_bbox(
    foreground: np.ndarray,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    ys, xs = np.where(foreground)
    if not xs.size:
        raise RuntimeError("Background removal found no foreground pixels in this image.")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    padding = int(round(max(x1 - x0, y1 - y0) * padding_ratio))
    height, width = foreground.shape
    return (
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(width, x1 + padding),
        min(height, y1 + padding),
    )


def _bbox_fill_fraction(foreground: np.ndarray) -> float:
    ys, xs = np.where(foreground)
    if not xs.size:
        return 0.0
    area = (int(xs.max()) - int(xs.min()) + 1) * (int(ys.max()) - int(ys.min()) + 1)
    return float(foreground.sum() / area)


def _erode_mask(mask: np.ndarray, erosion_ratio: float) -> tuple[np.ndarray, int]:
    if erosion_ratio <= 0:
        return mask.copy(), 0
    ys, xs = np.where(mask)
    if not xs.size:
        return mask.copy(), 0
    object_span = min(int(xs.max()) - int(xs.min()) + 1, int(ys.max()) - int(ys.min()) + 1)
    radius = max(1, int(round(object_span * erosion_ratio)))
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    eroded = cv2.erode(
        mask.astype(np.uint8),
        kernel,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=(0.0,),
    ).astype(bool)
    return eroded, radius


def _central_principal_axis_band(mask: np.ndarray, fraction: float) -> np.ndarray:
    """Keep a central band along the object's major axis to reject caps and end faces."""
    if fraction >= 1:
        return mask.copy()
    ys, xs = np.where(mask)
    if xs.size < 3:
        return mask.copy()
    coordinates = np.column_stack((xs, ys)).astype(np.float64)
    center = coordinates.mean(axis=0)
    covariance = np.cov(coordinates - center, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    projections = (coordinates - center) @ major_axis
    low, high = np.percentile(projections, (1, 99))
    half_width = max(float(high - low) * fraction / 2, 1.0)
    yy, xx = np.indices(mask.shape)
    all_projections = (xx - center[0]) * major_axis[0] + (yy - center[1]) * major_axis[1]
    return mask & (np.abs(all_projections - (low + high) / 2) <= half_width)


def _uint8_map(values: np.ndarray, opacity: np.ndarray) -> np.ndarray:
    return (np.clip(values, 0, 1) * 255 * opacity).astype(np.uint8)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "image"


def _image_sha256(image: Image.Image) -> str:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def _checkerboard(height: int, width: int, cell: int = 16) -> np.ndarray:
    ys, xs = np.mgrid[0:height, 0:width]
    values = (((ys // cell) + (xs // cell)) % 2 * 55 + 180).astype(np.uint8)
    return np.repeat(values[..., None], 3, axis=2)


def _foreground_preview(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    opacity = (alpha.astype(np.float32) / 255.0)[..., None]
    return (rgb * opacity + _checkerboard(*alpha.shape) * (1 - opacity)).astype(np.uint8)


def run_marigold(
    analyzer: MarigoldAnalyzer,
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
    scoring_mask_source: Image.Image | None = None,
    scoring_mask_rationale: str | None = None,
    run_key: str | None = None,
) -> dict[str, Any]:
    """Crop to the foreground, run Marigold, and persist compact diagnostics.

    ``run_key`` selects a stable output directory. Reusing it updates that run in
    place; omitting it retains timestamped history behavior for batch scripts.
    """
    if not 0 <= alpha_threshold <= 255:
        raise ValueError("alpha_threshold must be between 0 and 255")
    if not 0 <= crop_padding_ratio <= 1:
        raise ValueError("crop_padding_ratio must be between 0 and 1")
    if not 0 < contact_band_fraction <= 1:
        raise ValueError("contact_band_fraction must be between 0 and 1")
    if not 0 <= mask_erosion_ratio <= 0.25:
        raise ValueError("mask_erosion_ratio must be between 0 and 0.25")
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
    inference_rgb = input_rgb.crop(crop_bbox)
    inference_alpha = alpha[y0:y1, x0:x1]
    result = analyzer.analyze(
        inference_rgb,
        num_inference_steps=num_inference_steps,
        ensemble_size=ensemble_size,
        seed=seed,
    )
    map_height, map_width = result.roughness.shape
    alpha_small = np.asarray(
        Image.fromarray(inference_alpha, mode="L").resize(
            (map_width, map_height),
            resample=Image.Resampling.BILINEAR,
        ),
        dtype=np.uint8,
    )
    foreground = alpha_small >= alpha_threshold
    if not foreground.any():
        raise RuntimeError("Background removal found no foreground pixels in this image.")
    warnings: list[str] = []
    bbox_fill_fraction = _bbox_fill_fraction(source_foreground)
    if background_remover is None:
        warnings.append("background_removal_disabled")
    if bbox_fill_fraction < MIN_MASK_BBOX_FILL_FRACTION:
        warnings.append("sparse_foreground_mask")
    manual_outside_foreground_fraction = 0.0
    if scoring_mask_source is not None:
        scoring_mask = (
            np.asarray(
                scoring_mask_source.convert("L").resize(
                    (map_width, map_height),
                    resample=Image.Resampling.NEAREST,
                ),
                dtype=np.uint8,
            )
            >= 128
        )
        if int(scoring_mask.sum()) < MIN_SCORING_PIXELS:
            raise RuntimeError("The supplied grasp-contact mask is too small for analysis.")
        outside = scoring_mask & ~foreground
        manual_outside_foreground_fraction = float(outside.sum() / scoring_mask.sum())
        if manual_outside_foreground_fraction > 0.01:
            warnings.append("manual_mask_extends_beyond_automatic_foreground")
        # Transparent and reflective objects can be missed by automatic alpha matting.
        # A reviewed manual contact patch is authoritative for analysis, so retain it
        # in the analysis foreground while recording the disagreement above.
        foreground = foreground | scoring_mask
        erosion_radius = 0
        scoring_strategy = "manual_projected_gripper_contact"
    else:
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
        scoring_strategy = "eroded_central_principal_axis_band"

    opacity = alpha_small.astype(np.float32) / 255.0
    analysis_opacity = np.maximum(opacity, scoring_mask.astype(np.float32))
    albedo_display = (result.albedo_rgb * opacity[..., None]).astype(np.uint8)
    roughness_display = _uint8_map(result.roughness, analysis_opacity)
    metallicity_display = _uint8_map(result.metallicity, analysis_opacity)
    roughness_uncertainty_display = (
        _uint8_map(result.roughness_uncertainty, analysis_opacity)
        if result.roughness_uncertainty is not None
        else None
    )

    created_at = datetime.now(UTC)
    run_id = _slug(run_key) if run_key is not None else (
        f"{created_at:%Y%m%dT%H%M%S%fZ}_{_slug(source_label)}"
    )
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=run_key is not None)
    input_rgb.save(run_dir / "input.png")
    Image.fromarray(cutout_rgba, mode="RGBA").save(run_dir / "cutout.png")
    Image.fromarray(alpha, mode="L").save(run_dir / "foreground_mask.png")
    Image.fromarray(_foreground_preview(input_array, alpha)).save(
        run_dir / "foreground_preview.png"
    )
    inference_rgb.save(run_dir / "inference_crop.png")
    Image.fromarray(inference_alpha, mode="L").save(run_dir / "inference_mask.png")
    Image.fromarray(foreground.astype(np.uint8) * 255, mode="L").save(
        run_dir / "analysis_foreground_mask.png"
    )
    Image.fromarray(scoring_mask.astype(np.uint8) * 255, mode="L").save(
        run_dir / "scoring_mask.png"
    )
    Image.fromarray(albedo_display).save(run_dir / "albedo.png")
    Image.fromarray(roughness_display, mode="L").save(run_dir / "roughness.png")
    Image.fromarray(metallicity_display, mode="L").save(run_dir / "metallicity.png")
    if roughness_uncertainty_display is not None:
        Image.fromarray(roughness_uncertainty_display, mode="L").save(
            run_dir / "roughness_uncertainty.png"
        )

    roughness_uncertainty = (
        _statistics(result.roughness_uncertainty[scoring_mask])
        if result.roughness_uncertainty is not None
        else None
    )
    metallicity_uncertainty = (
        _statistics(result.metallicity_uncertainty[scoring_mask])
        if result.metallicity_uncertainty is not None
        else None
    )

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": created_at.isoformat(),
        "duration_seconds": round((created_at - started_at).total_seconds(), 3),
        "source": {
            "label": source_label,
            "dataset_id": dataset_id,
            "object_id": object_id,
            "path": source_path,
            "image_sha256": _image_sha256(input_rgb),
            "original_size": list(input_rgb.size),
        },
        "model": {
            "id": analyzer.model_id,
            "device": analyzer.device,
            "processing_resolution": analyzer.processing_resolution,
            "processed_size": list(result.processed_size),
            "num_inference_steps": num_inference_steps,
            "ensemble_size": ensemble_size,
            "seed": seed,
        },
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
            "size": list(inference_rgb.size),
            "foreground_fraction": float(np.mean(foreground)),
        },
        "scoring": {
            "strategy": scoring_strategy,
            "contact_band_fraction": contact_band_fraction,
            "mask_erosion_ratio": mask_erosion_ratio,
            "erosion_radius_pixels": erosion_radius,
            "foreground_pixels": int(foreground.sum()),
            "scoring_pixels": int(scoring_mask.sum()),
            "scoring_fraction_of_foreground": float(scoring_mask.sum() / foreground.sum()),
            "manual_mask_outside_automatic_foreground_fraction": (
                manual_outside_foreground_fraction
            ),
            "rationale": scoring_mask_rationale,
        },
        "quality": {
            "status": "warning" if warnings else "ok",
            "warnings": warnings,
            "interpretation": "BRDF appearance roughness; not physical height roughness or friction",
        },
        "roughness": _statistics(result.roughness[scoring_mask]),
        "metallicity": _statistics(result.metallicity[scoring_mask]),
        "roughness_uncertainty": roughness_uncertainty,
        "metallicity_uncertainty": metallicity_uncertainty,
        "artifacts": {
            "input": "input.png",
            "foreground_preview": "foreground_preview.png",
            "cutout": "cutout.png",
            "foreground_mask": "foreground_mask.png",
            "inference_crop": "inference_crop.png",
            "inference_mask": "inference_mask.png",
            "analysis_foreground_mask": "analysis_foreground_mask.png",
            "scoring_mask": "scoring_mask.png",
            "albedo": "albedo.png",
            "roughness": "roughness.png",
            "metallicity": "metallicity.png",
        },
    }
    if roughness_uncertainty_display is not None:
        metadata["artifacts"]["roughness_uncertainty"] = "roughness_uncertainty.png"
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    metadata["run_dir"] = str(run_dir)
    return metadata


def list_saved_runs(output_root: Path) -> list[dict[str, Any]]:
    """Load valid saved-run metadata, newest first."""
    runs: list[dict[str, Any]] = []
    if not output_root.is_dir():
        return runs
    for metadata_path in output_root.glob("*/metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(metadata, dict):
            continue
        metadata["run_dir"] = str(metadata_path.parent)
        runs.append(metadata)
    return sorted(runs, key=lambda item: str(item.get("created_at", "")), reverse=True)
