"""Perception: visual-semantic description + projected-contact-fraction proxy.

Two responsibilities, both pure/independent of hardware capture:
  1. describe(image)      -> short semantic description + surface material/condition
     (via the shared Gemini client; deterministic stub under dry_run).
  2. contact fraction     -> a = min(1, h_available / h_pad), the MVP height-ratio
     proxy. `contact_fraction_from_height_mm` is the honest, unit-tested core;
     `projected_contact_fraction` derives h_available from an Astra+ depth frame
     using the near-object vertical span (see docs/backlog.md for the planned
     segmentation/curvature upgrade).
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from .config import Config
from .models.gemini import get_client


class Description(BaseModel):
    retrieval_description: str = ""
    contact_region: str = "unknown"
    contact_material: str = "unknown"
    visible_surface_material: str = "unknown"
    visible_surface_condition: str = "unknown"
    local_geometry: str = "unknown"
    contact_patch_visibility: str = "unknown"
    uncertainty: str = "unknown"

    @property
    def description(self) -> str:
        """Backward-compatible semantic text consumed by retrieval."""
        return self.retrieval_description


def describe(image_bgr: np.ndarray | None, cfg: Config) -> Description:
    """Visual-semantic description of an object for the grasping database."""
    if cfg.models.dry_run or image_bgr is None:
        return Description(retrieval_description="synthetic object (dry-run)")
    raw = get_client(cfg).generate_json(
        system=cfg.prompts.descriptor_system,
        instruction=cfg.prompts.descriptor,
        schema=Description,
        image_bgr=image_bgr,
    )
    return Description.model_validate(raw)


# --------------------------------------------------------------------------- #
# Projected contact fraction (MVP height-ratio proxy)
# --------------------------------------------------------------------------- #
def contact_fraction_from_height_mm(h_available_mm: float, cfg: Config) -> float:
    """a = min(1, h_available / h_pad). Honest core used by tests and callers."""
    h_pad = cfg.geometry.pad_height_mm
    if h_pad <= 0:
        raise ValueError("pad_height_mm must be positive")
    return float(min(1.0, max(0.0, h_available_mm / h_pad)))


def near_object_mask(depth_mm: np.ndarray, near_band_mm: float = 150.0) -> np.ndarray:
    """Pixels belonging to the closest object cluster (valid & near the minimum)."""
    valid = np.isfinite(depth_mm) & (depth_mm >= 20.0) & (depth_mm <= 10000.0)
    if not valid.any():
        return np.zeros_like(depth_mm, dtype=bool)
    nearest = float(depth_mm[valid].min())
    return valid & (depth_mm <= nearest + near_band_mm)


def projected_contact_fraction(depth_mm: np.ndarray, cfg: Config) -> float:
    """Estimate contact fraction from the near-object vertical extent in a depth frame.

    MVP assumption: the scene is framed so that a fully pad-height object spans the
    frame; the object's vertical pixel span therefore approximates its coverage of
    the pad. Returns a in [0, 1].
    """
    mask = near_object_mask(depth_mm)
    if not mask.any():
        return 0.0
    rows = np.where(mask.any(axis=1))[0]
    span_px = int(rows.max() - rows.min() + 1)
    h_available_mm = (span_px / depth_mm.shape[0]) * cfg.geometry.pad_height_mm
    return contact_fraction_from_height_mm(h_available_mm, cfg)
