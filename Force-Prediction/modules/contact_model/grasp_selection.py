"""Select the physical first-touch anchors for a parallel-jaw grasp."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contact_geometry import Boundary


@dataclass
class GraspPair:
    left_anchor: int
    right_anchor: int
    antipodal: bool
    left_center: int
    right_center: int
    score: float


def side_alignment(b: Boundary, side: str) -> np.ndarray:
    """Cosine alignment of each outward normal with its jaw direction."""
    side_sign = -1.0 if side == "left" else 1.0
    return side_sign * b.N[:, 0]


def anchor_in_band(
    b: Boundary,
    band_idx: np.ndarray,
    side: str,
    center_y: float,
    side_angle_deg: float,
    reachable: np.ndarray,
    seat_tolerance_mm: float = 1.0,
) -> int:
    """Return the outwardmost side-facing sample in the active pad band.

    Flat-face spline overshoot is suppressed when the raw extremum is within
    ``seat_tolerance_mm`` of a reachable side sample. A genuinely protruding
    nonconformable feature still wins and produces a zero-length patch
    downstream. Ties within 0.05 mm resolve toward the pad-window centre.
    """
    if not 0.0 < side_angle_deg < 90.0:
        raise ValueError("side_angle_deg must be between 0 and 90 degrees")

    threshold = float(np.cos(np.radians(side_angle_deg)))
    eligible = band_idx[side_alignment(b, side)[band_idx] >= threshold]
    if len(eligible) == 0:
        return -1

    side_sign = -1.0 if side == "left" else 1.0
    projection = side_sign * b.pts[eligible, 0]
    extremum = float(projection.max())
    reachable_local = reachable[eligible]
    if reachable_local.any():
        reachable_extremum = float(projection[reachable_local].max())
        if extremum - reachable_extremum <= seat_tolerance_mm:
            ties = eligible[
                reachable_local & (projection >= reachable_extremum - 0.05)
            ]
            return int(ties[np.argmin(np.abs(b.pts[ties, 1] - center_y))])

    ties = eligible[projection >= extremum - 0.05]
    return int(ties[np.argmin(np.abs(b.pts[ties, 1] - center_y))])


def pair_from_anchors(
    b: Boundary, left_anchor: int, right_anchor: int,
    antipodal_tolerance_deg: float = 40.0,
) -> GraspPair:
    """Build a pair and apply the existing opposed-normal criterion."""
    if left_anchor < 0 or right_anchor < 0:
        return GraspPair(
            left_anchor, right_anchor, False,
            left_anchor, right_anchor, 0.0,
        )
    score = -float(b.N[left_anchor] @ b.N[right_anchor])
    antipodal = bool(score >= np.cos(np.radians(antipodal_tolerance_deg)))
    return GraspPair(
        left_anchor, right_anchor, antipodal,
        left_anchor, right_anchor, score,
    )
