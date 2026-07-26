"""Projected two-pad contact fraction from a 2D object silhouette.

The camera observes the longitudinal plane in which the parallel jaws close.
Pad width is assumed constant and fully available wherever longitudinal
contact exists, so it cancels from the required ratio:

    geometric_fraction = (left_contact_length + right_contact_length)
                         / (2 * pad_length)
    combined_fraction = max(minimum_contact_fraction, geometric_fraction)

This is a macroscopic projected-contact proxy, not microscopic adhesive area.
Each pad produces one contiguous patch from its first-touch anchor.  Traversal
stops permanently when pad budget, side orientation, convex curvature, or
concave accessibility fails.

For an otherwise valid antipodal grasp, a configurable minimum fraction
represents the unavoidable finite seating/contact patch that is not resolved
by the macroscopic boundary walk. Rejected non-antipodal grasps remain zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contact_geometry import Boundary, build_boundary
from .finger_drape import reachability_mask
from .grasp_selection import (
    GraspPair,
    anchor_in_band,
    pair_from_anchors,
    side_alignment,
)


@dataclass
class FingerResult:
    side: str
    anchor: int
    contact_idx: np.ndarray
    contact_points: np.ndarray
    contact_length: float
    pad_length: float
    fraction: float


@dataclass
class ContactEstimate:
    boundary: Boundary
    reachable: np.ndarray
    pair: GraspPair
    left: FingerResult
    right: FingerResult
    pad_length_mm: float
    minimum_bend_radius_mm: float
    side_angle_deg: float
    minimum_contact_fraction: float
    pad_band: tuple[float, float]
    feasible: bool

    @property
    def k_max_per_mm(self) -> float:
        return 1.0 / self.minimum_bend_radius_mm

    @property
    def combined_contact_length(self) -> float:
        """Geometrically resolved green-path length, before the floor."""
        return min(
            self.left.contact_length + self.right.contact_length,
            2.0 * self.pad_length_mm,
        )

    @property
    def geometric_contact_fraction(self) -> float:
        return float(np.clip(
            self.combined_contact_length / (2.0 * self.pad_length_mm),
            0.0,
            1.0,
        ))

    @property
    def contact_floor_applied(self) -> bool:
        return bool(
            self.pair.antipodal
            and self.geometric_contact_fraction < self.minimum_contact_fraction
        )

    @property
    def combined_contact_fraction(self) -> float:
        """Authoritative ratio, including the minimum physical-contact floor."""
        if not self.pair.antipodal:
            return 0.0
        return max(
            self.geometric_contact_fraction,
            self.minimum_contact_fraction,
        )


def _empty_finger(side: str, pad_length_mm: float, anchor: int = -1) -> FingerResult:
    return FingerResult(
        side=side,
        anchor=anchor,
        contact_idx=np.zeros(0, dtype=int),
        contact_points=np.empty((0, 2), dtype=float),
        contact_length=0.0,
        pad_length=pad_length_mm,
        fraction=0.0,
    )


def _sample_is_contactable(
    b: Boundary,
    idx: int,
    side: str,
    reachable: np.ndarray,
    side_threshold: float,
    pad_lo: float,
    pad_hi: float,
) -> bool:
    y = float(b.pts[idx, 1])
    return bool(
        pad_lo - 1e-9 <= y <= pad_hi + 1e-9
        and reachable[idx]
        and side_alignment(b, side)[idx] >= side_threshold
    )


def _walk_direction(
    b: Boundary,
    anchor: int,
    direction: int,
    budget_mm: float,
    side: str,
    reachable: np.ndarray,
    side_threshold: float,
    pad_lo: float,
    pad_hi: float,
) -> tuple[list[int], list[np.ndarray], float]:
    """Walk one contiguous side patch and integrate true segment lengths."""
    indices: list[int] = []
    points: list[np.ndarray] = []
    length = 0.0
    current = anchor
    n = len(b)

    for _ in range(n // 2):
        remaining = budget_mm - length
        if remaining <= 1e-12:
            break
        nxt = (current + direction) % n
        if not _sample_is_contactable(
            b, nxt, side, reachable, side_threshold, pad_lo, pad_hi
        ):
            break

        segment = float(np.linalg.norm(b.pts[nxt] - b.pts[current]))
        if segment <= 1e-12:
            current = nxt
            continue
        if segment <= remaining + 1e-12:
            length += segment
            indices.append(nxt)
            points.append(b.pts[nxt].copy())
            current = nxt
            continue

        # The material budget ends inside this boundary segment.  Integrate
        # the exact remaining length and interpolate the plotted endpoint.
        fraction = remaining / segment
        points.append(b.pts[current] + fraction * (b.pts[nxt] - b.pts[current]))
        length += remaining
        break

    return indices, points, length


def _analyze_finger(
    b: Boundary,
    anchor: int,
    side: str,
    reachable: np.ndarray,
    pad_lo: float,
    pad_hi: float,
    pad_length_mm: float,
    side_angle_deg: float,
) -> FingerResult:
    if anchor < 0:
        return _empty_finger(side, pad_length_mm)

    threshold = float(np.cos(np.radians(side_angle_deg)))
    if not _sample_is_contactable(
        b, anchor, side, reachable, threshold, pad_lo, pad_hi
    ):
        return _empty_finger(side, pad_length_mm, anchor)

    anchor_y = float(b.pts[anchor, 1])
    walks: dict[int, tuple[list[int], list[np.ndarray], float]] = {}
    for direction in (-1, +1):
        probe = (anchor + direction) % len(b)
        moves_up = b.pts[probe, 1] >= anchor_y
        budget = pad_hi - anchor_y if moves_up else anchor_y - pad_lo
        walks[direction] = _walk_direction(
            b, anchor, direction, max(0.0, float(budget)), side,
            reachable, threshold, pad_lo, pad_hi,
        )

    neg_idx, neg_points, neg_length = walks[-1]
    pos_idx, pos_points, pos_length = walks[+1]
    contact_length = min(neg_length + pos_length, pad_length_mm)
    ordered_idx = np.array(
        list(reversed(neg_idx)) + [anchor] + pos_idx, dtype=int
    )
    ordered_points = np.vstack(
        list(reversed(neg_points)) + [b.pts[anchor].copy()] + pos_points
    )
    return FingerResult(
        side=side,
        anchor=anchor,
        contact_idx=ordered_idx,
        contact_points=ordered_points,
        contact_length=contact_length,
        pad_length=pad_length_mm,
        fraction=float(np.clip(contact_length / pad_length_mm, 0.0, 1.0)),
    )


def estimate_contact(
    pts_mm: np.ndarray,
    *,
    pad_length_mm: float = 106.68,
    minimum_bend_radius_mm: float = 20.0,
    side_angle_deg: float = 30.0,
    minimum_contact_fraction: float = 0.05,
    ds: float = 0.25,
    smoothing_mm: float = 0.2,
) -> ContactEstimate:
    """Estimate contiguous projected side contact for both gripper pads."""
    if pad_length_mm <= 0:
        raise ValueError("pad_length_mm must be positive")
    if minimum_bend_radius_mm <= 0:
        raise ValueError("minimum_bend_radius_mm must be positive")
    if not 0.0 < side_angle_deg < 90.0:
        raise ValueError("side_angle_deg must be between 0 and 90 degrees")
    if not 0.0 <= minimum_contact_fraction <= 1.0:
        raise ValueError("minimum_contact_fraction must be between 0 and 1")

    b = build_boundary(pts_mm, ds=ds, smoothing_mm=smoothing_mm)
    reachable = reachability_mask(b, minimum_bend_radius_mm)
    pad_hi = float(b.pts[:, 1].max())
    pad_lo = pad_hi - pad_length_mm
    band_idx = np.where(
        (b.pts[:, 1] >= pad_lo - 1e-9) & (b.pts[:, 1] <= pad_hi + 1e-9)
    )[0]
    band_center = 0.5 * (pad_lo + pad_hi)

    left_anchor = anchor_in_band(
        b, band_idx, "left", band_center, side_angle_deg, reachable
    )
    right_anchor = anchor_in_band(
        b, band_idx, "right", band_center, side_angle_deg, reachable
    )
    pair = pair_from_anchors(b, left_anchor, right_anchor)

    if not pair.antipodal:
        left = _empty_finger("left", pad_length_mm, left_anchor)
        right = _empty_finger("right", pad_length_mm, right_anchor)
        return ContactEstimate(
            b, reachable, pair, left, right, pad_length_mm,
            minimum_bend_radius_mm, side_angle_deg, minimum_contact_fraction,
            (pad_lo, pad_hi), False,
        )

    left = _analyze_finger(
        b, left_anchor, "left", reachable, pad_lo, pad_hi,
        pad_length_mm, side_angle_deg,
    )
    right = _analyze_finger(
        b, right_anchor, "right", reachable, pad_lo, pad_hi,
        pad_length_mm, side_angle_deg,
    )
    feasible = bool(
        left.contact_length > 0.0
        or right.contact_length > 0.0
        or minimum_contact_fraction > 0.0
    )
    return ContactEstimate(
        b, reachable, pair, left, right, pad_length_mm,
        minimum_bend_radius_mm, side_angle_deg, minimum_contact_fraction,
        (pad_lo, pad_hi), feasible,
    )
