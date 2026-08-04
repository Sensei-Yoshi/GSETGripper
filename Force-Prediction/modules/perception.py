"""Visual-semantic object perception for retrieval and prediction."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from .config import Config
from .models.gemini import get_client


class Description(BaseModel):
    retrieval_description: str = ""
    contact_material: str = "unknown"
    visible_surface_material: str = "unknown"
    visible_surface_condition: str = "unknown"
    contact_patch_visibility: str = "unknown"
    uncertainty: str = "unknown"

def describe(image_bgr: np.ndarray | None, cfg: Config) -> Description:
    """Visual-semantic description of an object for the grasping database."""
    if image_bgr is None:
        raise ValueError("Gemini description requires a decodable object image")
    raw = get_client(cfg).generate_json(
        system=cfg.prompts.descriptor_system,
        instruction=cfg.prompts.descriptor,
        schema=Description,
        image_bgr=image_bgr,
    )
    return Description.model_validate(raw)
