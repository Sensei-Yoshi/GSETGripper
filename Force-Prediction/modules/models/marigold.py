"""Lazy Marigold IID inference and self-contained roughness-run persistence.

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

import numpy as np
from PIL import Image

from .background_remover import BackgroundRemover

DEFAULT_MODEL_ID = "prs-eth/marigold-iid-appearance-v1-1"
SCHEMA_VERSION = 2
DEFAULT_ALPHA_THRESHOLD = 120


@dataclass(frozen=True)
class IntrinsicPrediction:
    """Marigold appearance outputs at the model processing resolution."""

    albedo_rgb: np.ndarray
    roughness: np.ndarray
    metallicity: np.ndarray
    processed_size: tuple[int, int]


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
        processing_resolution: int = 640,
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
        num_inference_steps: int = 1,
        seed: int = 2024,
    ) -> IntrinsicPrediction:
        """Decompose an RGB image into albedo, roughness, and metallicity."""
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        if self.pipe is None:
            self._load()
        assert self.pipe is not None

        import torch

        prepared = image.convert("RGB")
        prepared.thumbnail((self.processing_resolution, self.processing_resolution))
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipe(
            prepared,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        prediction = np.asarray(result.prediction)
        properties = self.pipe.target_properties
        material_index = properties["target_names"].index("material")
        material_names = properties["material"]["sub_target_names"]
        material = prediction[material_index]
        roughness = np.squeeze(material[..., material_names.index("roughness")]).astype(
            np.float32
        )
        metallicity = np.squeeze(material[..., material_names.index("metallicity")]).astype(
            np.float32
        )

        visualized = self.pipe.image_processor.visualize_intrinsics(
            result.prediction, properties
        )
        albedo = np.asarray(visualized[0]["albedo"].convert("RGB"), dtype=np.uint8)
        if self.device == "mps":
            torch.mps.empty_cache()
        return IntrinsicPrediction(
            albedo_rgb=albedo,
            roughness=roughness,
            metallicity=metallicity,
            processed_size=prepared.size,
        )


def _statistics(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {key: float("nan") for key in ("mean", "median", "std", "min", "max", "p05", "p95")}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
    }


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
    num_inference_steps: int = 1,
    seed: int = 2024,
) -> dict[str, Any]:
    """Remove the background, run Marigold, and persist all diagnostic artifacts."""
    if not 0 <= alpha_threshold <= 255:
        raise ValueError("alpha_threshold must be between 0 and 255")
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
    result = analyzer.analyze(
        input_rgb,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )
    map_height, map_width = result.roughness.shape
    alpha_small = np.asarray(
        Image.fromarray(alpha, mode="L").resize(
            (map_width, map_height),
            resample=Image.Resampling.BILINEAR,
        ),
        dtype=np.uint8,
    )
    foreground = alpha_small >= alpha_threshold
    if not foreground.any():
        raise RuntimeError("Background removal found no foreground pixels in this image.")
    opacity = alpha_small.astype(np.float32) / 255.0
    albedo_display = (result.albedo_rgb * opacity[..., None]).astype(np.uint8)
    roughness_display = (
        np.clip(result.roughness, 0, 1) * 255 * opacity
    ).astype(np.uint8)
    metallicity_display = (
        np.clip(result.metallicity, 0, 1) * 255 * opacity
    ).astype(np.uint8)

    created_at = datetime.now(UTC)
    run_id = f"{created_at:%Y%m%dT%H%M%S%fZ}_{_slug(source_label)}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    input_rgb.save(run_dir / "input.png")
    Image.fromarray(cutout_rgba, mode="RGBA").save(run_dir / "cutout.png")
    Image.fromarray(alpha, mode="L").save(run_dir / "foreground_mask.png")
    Image.fromarray(_foreground_preview(input_array, alpha)).save(
        run_dir / "foreground_preview.png"
    )
    Image.fromarray(albedo_display).save(run_dir / "albedo.png")
    Image.fromarray(roughness_display, mode="L").save(run_dir / "roughness.png")
    Image.fromarray(metallicity_display, mode="L").save(run_dir / "metallicity.png")
    np.save(run_dir / "roughness.npy", result.roughness)
    np.save(run_dir / "metallicity.npy", result.metallicity)
    np.save(run_dir / "foreground_mask.npy", foreground)

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
            "seed": seed,
        },
        "background_removal": {
            "enabled": background_remover is not None,
            "model": background_model,
            "alpha_threshold": alpha_threshold,
            "foreground_fraction": float(np.mean(foreground)),
        },
        "roughness": _statistics(result.roughness[foreground]),
        "metallicity": _statistics(result.metallicity[foreground]),
        "artifacts": {
            "input": "input.png",
            "foreground_preview": "foreground_preview.png",
            "cutout": "cutout.png",
            "foreground_mask": "foreground_mask.png",
            "albedo": "albedo.png",
            "roughness": "roughness.png",
            "metallicity": "metallicity.png",
            "roughness_data": "roughness.npy",
            "metallicity_data": "metallicity.npy",
            "foreground_mask_data": "foreground_mask.npy",
        },
    }
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
