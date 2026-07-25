"""Modules 3+4 - finger reachability and the draping walk.

Reachability (can the pad conform locally at boundary point i?):

  1. Convex conformity: kappa[i] <= k_max. A tighter bulge than the finger's
     minimum bend radius r_min = 1/k_max cannot be hugged.
  2. Concave fit: the external disk of radius r_min tangent at i (centre
     p + r_min * N) must contain no other boundary point. This is the
     rolling-ball / morphological-closing test and catches waists whose
     mouth or floor is too tight, both locally and globally (KD-tree).

  Note the two tests are complementary: a rolling external disk touches
  every point of a *convex* region regardless of curvature, so the closing
  operation alone cannot represent the convex failure case - hence test 1.

Draping walk (per finger, from the anchor outward in both directions):

  - While in CONTACT: advance point by point as long as the point is
    reachable. The anchor itself always counts as contact (the jaw presses
    there).
  - On hitting an unreachable point the finger DEPARTS on its max-wrap arc:
    radius r_min, tangent-continuous at the departure point, curving toward
    the object (centre c = p - r_min * N). This is the best case the
    structure allows and is capped at ARC_CAP_RAD of wrap.
  - While AIRBORNE: the gap to the surface is g = r_min - |b - c| (the
    boundary sits inside the wrap circle). g <= delta marks tolerance
    contact (the Hertz-like delta-fringe falls out of the exact arc
    geometry). g <= 0 means the surface rises back through the finger's
    path: the finger RE-LANDS there and the walk resumes in CONTACT.

All quantities are exact circle geometry - the quadratic small-gap formulas
used for the analytic ground truths are approximations of *this*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from contact_geometry import Boundary

ARC_CAP_RAD = 2.0  # max wrap angle followed on one departure arc (~115 deg)


@dataclass
class WrapArc:
    center: np.ndarray
    r: float
    phi0: float
    sweep: float          # +1 CCW / -1 CW as seen in the xy plane
    start_idx: int
    max_phi: float = field(default=0.3)  # widest angle reached (for plotting)


def reachability_mask(b: Boundary, k_max: float) -> np.ndarray:
    r_min = 1.0 / k_max
    convex_ok = b.kappa <= k_max + 1e-9
    centers = b.pts + r_min * b.N
    tree = cKDTree(b.pts)
    hits = tree.query_ball_point(centers, r_min * (1.0 - 1e-6))
    disk_ok = np.fromiter((len(h) == 0 for h in hits), dtype=bool, count=len(b))
    return convex_ok & disk_ok


def _make_arc(b: Boundary, d_idx: int, direction: int, r_min: float) -> WrapArc:
    p = b.pts[d_idx]
    t_walk = direction * b.T[d_idx]
    center = p - r_min * b.N[d_idx]
    v = p - center
    phi0 = float(np.arctan2(v[1], v[0]))
    cp = center - p
    sweep = float(np.sign(t_walk[0] * cp[1] - t_walk[1] * cp[0])) or 1.0
    return WrapArc(center, r_min, phi0, sweep, d_idx)


def _gap_to_arc(arc: WrapArc, p: np.ndarray) -> tuple[float, float]:
    v = p - arc.center
    dist = float(np.hypot(v[0], v[1]))
    dphi = (arc.sweep * (np.arctan2(v[1], v[0]) - arc.phi0)) % (2.0 * np.pi)
    if dphi > ARC_CAP_RAD:
        return np.inf, dphi
    return arc.r - dist, dphi


def drape(
    b: Boundary,
    anchor: int,
    direction: int,
    max_steps: int,
    k_max: float,
    delta: float,
    reachable: np.ndarray,
    side_sign: float = 0.0,
    y_min: float | None = None,
) -> tuple[dict[int, tuple[bool, float]], list[WrapArc]]:
    """Walk ``max_steps`` samples from ``anchor`` in ``direction`` (+1/-1).

    ``side_sign`` (+1 right finger, -1 left) stops the walk once the surface
    normal turns away from the closing direction (pad wrapped past a pole -
    the straight-backboned finger cannot follow over the top). ``y_min``
    stops it below the fingertip height (nothing exists below the tip).
    Returns {boundary index: (in_contact, gap_mm)} and the wrap arcs taken.
    """
    n = len(b)
    r_min = 1.0 / k_max
    record: dict[int, tuple[bool, float]] = {}
    arcs: list[WrapArc] = []
    state = "contact"
    last = anchor
    arc: WrapArc | None = None

    for step in range(max_steps + 1):
        idx = (anchor + direction * step) % n
        if step > 0:
            if y_min is not None and b.pts[idx, 1] < y_min:
                break
            if side_sign and side_sign * b.N[idx, 0] < -0.1:
                break
        if state == "contact":
            if step == 0 or reachable[idx]:
                record[idx] = (True, 0.0)
                last = idx
                if step == 0 and not reachable[idx]:
                    arc = _make_arc(b, last, direction, r_min)
                    arcs.append(arc)
                    state = "air"
                continue
            arc = _make_arc(b, last, direction, r_min)
            arcs.append(arc)
            state = "air"

        assert arc is not None
        gap, dphi = _gap_to_arc(arc, b.pts[idx])
        if np.isfinite(gap):
            arc.max_phi = max(arc.max_phi, min(dphi, ARC_CAP_RAD))
        if gap <= 1e-9:  # surface rises through the finger path: re-land
            record[idx] = (True, 0.0)
            last = idx
            state = "contact"
        else:
            record[idx] = (gap <= delta, gap)

    return record, arcs
