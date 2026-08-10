"""Dataset-relative roughness representations used in VLM evidence payloads."""

from __future__ import annotations

from typing import Literal

RoughnessCategory = Literal["smooth", "rough"]


def binary_roughness_category(
    roughness_index: float,
    threshold: float,
) -> RoughnessCategory:
    """Map a measured index to the configured dataset-relative binary class."""
    return "rough" if roughness_index >= threshold else "smooth"
