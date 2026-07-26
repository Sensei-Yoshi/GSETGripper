"""Projected two-pad contact estimation from calibrated RGB silhouettes."""

from .contact_area import ContactEstimate, FingerResult, estimate_contact
from .extract_object_outline import (
    REMBG_MODEL,
    create_rembg_session,
    largest_foreground_component,
    process_image,
    segment_foreground,
)
from .pipeline_core import (
    SUMMARY_SCHEMA_VERSION,
    ContactParams,
    analyze_image,
    build_summary,
    outline_csv_to_mm,
)

__all__ = [
    "SUMMARY_SCHEMA_VERSION",
    "ContactEstimate",
    "ContactParams",
    "FingerResult",
    "REMBG_MODEL",
    "analyze_image",
    "build_summary",
    "create_rembg_session",
    "estimate_contact",
    "largest_foreground_component",
    "outline_csv_to_mm",
    "process_image",
    "segment_foreground",
]
