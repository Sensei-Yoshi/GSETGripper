"""Module 5 + orchestration - transverse width, area, contact fraction.

Transverse (out-of-plane) effective width per contact point:

    w_eff = min(w_pad, 2 * sqrt(2 * R_t * delta))

with the *corrected* transverse radius for axisymmetric objects:

    R_t = r_parallel / |N_x|

where r_parallel is the horizontal distance from the symmetry axis and N_x
the horizontal component of the outward normal (Meusnier). For a sphere this
is exactly R everywhere; the naive r_parallel alone underestimates R_t away
from the equator. Prismatic objects use R_t = inf, i.e. w_eff = w_pad.

Contact fraction (per finger) = area / (window_length * w_pad).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contact_geometry import Boundary, build_boundary
from finger_drape import WrapArc, drape, reachability_mask
from grasp_selection import GraspPair, select_grasp_pair


@dataclass
class FingerResult:
    side: str
    anchor: int
    window_idx: np.ndarray   # boundary indices, ordered along the window
    contact: np.ndarray      # bool, aligned with window_idx
    gap: np.ndarray          # mm, aligned with window_idx
    w_eff: np.ndarray        # mm, aligned with window_idx
    arcs: list[WrapArc]
    contact_length: float    # mm
    window_length: float     # mm
    area: float              # mm^2
    fraction: float          # dimensionless


@dataclass
class ContactEstimate:
    boundary: Boundary
    reachable: np.ndarray
    pair: GraspPair
    left: FingerResult
    right: FingerResult
    k_max: float
    delta: float
    L: float
    w_pad: float
    object_type: str

    @property
    def total_area(self) -> float:
        return self.left.area + self.right.area

    @property
    def mean_fraction(self) -> float:
        return 0.5 * (self.left.fraction + self.right.fraction)


def transverse_width(
    b: Boundary,
    idx: np.ndarray,
    object_type: str,
    delta: float,
    w_pad: float,
    axis_x: float,
) -> np.ndarray:
    if object_type == "prismatic":
        return np.full(len(idx), w_pad)
    r_par = np.abs(b.pts[idx, 0] - axis_x)
    n_x = np.maximum(np.abs(b.N[idx, 0]), 0.15)
    r_t = np.minimum(r_par / n_x, 1e4)
    return np.minimum(w_pad, 2.0 * np.sqrt(2.0 * r_t * delta))


def _analyze_finger(
    b: Boundary,
    anchor: int,
    side: str,
    half_pts: int,
    k_max: float,
    delta: float,
    reachable: np.ndarray,
    w_pad: float,
    object_type: str,
    axis_x: float,
) -> FingerResult:
    record: dict[int, tuple[bool, float]] = {}
    arcs: list[WrapArc] = []
    for direction in (+1, -1):
        rec, a = drape(b, anchor, direction, half_pts, k_max, delta, reachable)
        for idx, (c, g) in rec.items():
            if idx in record:
                pc, pg = record[idx]
                record[idx] = (pc or c, min(pg, g))
            else:
                record[idx] = (c, g)
        arcs.extend(a)

    window_idx = (anchor + np.arange(-half_pts, half_pts + 1)) % len(b)
    contact = np.array([record.get(i, (False, np.inf))[0] for i in window_idx])
    gap = np.array([record.get(i, (False, np.inf))[1] for i in window_idx])
    w_eff = transverse_width(b, window_idx, object_type, delta, w_pad, axis_x)

    contact_length = float(contact.sum() * b.ds)
    window_length = float(len(window_idx) * b.ds)
    area = float(np.sum(w_eff[contact]) * b.ds)
    fraction = area / (window_length * w_pad)
    return FingerResult(
        side, anchor, window_idx, contact, gap, w_eff, arcs,
        contact_length, window_length, area, fraction,
    )


def estimate_contact(
    pts_mm: np.ndarray,
    k_max: float,
    delta: float = 0.3,
    L: float = 4.0,
    w_pad: float = 12.0,
    ds: float = 0.25,
    smoothing_mm: float = 0.0,
    object_type: str = "prismatic",
    y_target: float | None = None,
) -> ContactEstimate:
    """End-to-end estimate: boundary -> grasp -> drape -> area."""
    b = build_boundary(pts_mm, ds=ds, smoothing_mm=smoothing_mm)
    reachable = reachability_mask(b, k_max)
    pair = select_grasp_pair(b, L, reachable, y_target=y_target)
    half_pts = min(int(round((L / 2) / b.ds)), len(b) // 2 - 1)
    axis_x = float(b.pts[:, 0].mean())

    left = _analyze_finger(
        b, pair.left_anchor, "left", half_pts, k_max, delta,
        reachable, w_pad, object_type, axis_x,
    )
    right = _analyze_finger(
        b, pair.right_anchor, "right", half_pts, k_max, delta,
        reachable, w_pad, object_type, axis_x,
    )
    return ContactEstimate(
        b, reachable, pair, left, right, k_max, delta, L, w_pad, object_type
    )
