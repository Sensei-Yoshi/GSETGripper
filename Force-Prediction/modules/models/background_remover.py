"""Lazy reusable foreground matting backed by rembg."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

DEFAULT_BACKGROUND_MODEL = "isnet-general-use"


@dataclass(frozen=True)
class BackgroundRemoval:
    """Full-resolution RGBA cutout and its uint8 alpha matte."""

    cutout_rgba: np.ndarray
    alpha: np.ndarray


class BackgroundRemover:
    """Load a rembg session only when the first image is processed."""

    def __init__(self, model_name: str = DEFAULT_BACKGROUND_MODEL) -> None:
        self.model_name = model_name
        self.session: Any | None = None

    def _load(self) -> None:
        try:
            from rembg import new_session
        except ImportError as error:
            raise RuntimeError(
                "Background removal requires rembg. Install this project with "
                "`pip install -e '.[roughness]'`."
            ) from error
        self.session = new_session(self.model_name)

    def remove(self, image: Image.Image) -> BackgroundRemoval:
        """Remove the background and return a soft alpha matte."""
        if self.session is None:
            self._load()
        try:
            from rembg import remove
        except ImportError as error:
            raise RuntimeError(
                "Background removal requires rembg. Install this project with "
                "`pip install -e '.[roughness]'`."
            ) from error

        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        output = remove(rgb, session=self.session)
        if isinstance(output, Image.Image):
            rgba = np.asarray(output.convert("RGBA"), dtype=np.uint8)
        else:
            rgba = np.asarray(output, dtype=np.uint8)
        if rgba.ndim != 3 or rgba.shape[2] != 4:
            raise RuntimeError("Background remover did not return an RGBA image.")
        return BackgroundRemoval(cutout_rgba=rgba, alpha=rgba[..., 3])
