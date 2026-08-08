"""Experiment-ID compatibility across definition versions."""

from __future__ import annotations

GAPPED_EXPERIMENT_DEFINITION_VERSION = 12

# Definition v12 removed the former E2 while retaining the established E3-E6 IDs.
# Definition v13 closes that gap. Saved v12 artifacts remain immutable and are
# translated only while being loaded or displayed.
LEGACY_V12_TO_ACTIVE_EXPERIMENT_IDS = {
    "e1": "e1",
    "e3": "e2",
    "e4": "e3",
    "e5": "e4",
    "e6": "e5",
}


def resolve_experiment_id(
    experiment_id: str,
    experiment_definition_version: object = None,
) -> str:
    """Return the active ID for a stored experiment/version pair."""
    normalized = experiment_id.lower()
    if experiment_definition_version == GAPPED_EXPERIMENT_DEFINITION_VERSION:
        return LEGACY_V12_TO_ACTIVE_EXPERIMENT_IDS.get(normalized, normalized)
    return normalized


def is_legacy_v12_definition(experiment_definition_version: object) -> bool:
    """Whether a payload uses the one-off gapped v12 experiment namespace."""
    return experiment_definition_version == GAPPED_EXPERIMENT_DEFINITION_VERSION
