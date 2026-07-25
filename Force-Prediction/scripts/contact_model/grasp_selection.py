"""Module 2 - grasp anchor selection.

Jaws close along +/-x, so each finger's physical first touch is the
object's extremal-x *conformable* point - the widest point the pad can
seat on, NOT a preferred height. Exact ties (flat walls, where the whole
face touches simultaneously) break toward ``center_y`` (pad centre in
finger mode, ``y_target``/centroid in free mode). A true vertex that
protrudes more than ``seat_tol`` beyond every conformable point wins
anyway (pentagon corner: the jaw really does stop there).

Antipodality is *reported*, not enforced: parallel jaws close where the
geometry says they close, even on a face-vs-vertex pentagon grasp.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contact_geometry import Boundary


@dataclass
class GraspPair:
    left_anchor: int
    right_anchor: int
    antipodal: bool
    left_center: int
    right_center: int
    score: float


def anchor_in_band(
    b: Boundary,
    cand_idx: np.ndarray,
    side: str,
    reachable: np.ndarray,
    center_y: float,
    seat_tol: float = 1.0,
) -> int:
    """First-touch anchor within a candidate index set: the extremal-x
    conformable point (spline overshoot at sharp corners can make an
    unreachable corner the raw extremum by a hair - seating there would
    lose the adjacent flat), unless the raw extremum protrudes more than
    ``seat_tol`` beyond every conformable point (a true vertex). Ties
    break toward ``center_y``."""
    sgn = -1.0 if side == "left" else 1.0
    proj = sgn * b.pts[cand_idx, 0]
    ext = proj.max()

    reach_local = reachable[cand_idx]
    if reach_local.any():
        ext_r = proj[reach_local].max()
        if ext - ext_r <= seat_tol:
            ties = cand_idx[reach_local & (proj >= ext_r - 0.05)]
            return int(ties[np.argmin(np.abs(b.pts[ties, 1] - center_y))])

    ties = cand_idx[proj >= ext - 0.05]
    return int(ties[np.argmin(np.abs(b.pts[ties, 1] - center_y))])


def select_grasp_pair(
    b: Boundary,
    reachable: np.ndarray,
    y_target: float | None = None,
) -> GraspPair:
    """Free-placement anchors: the global extremal-x conformable point per
    side - the jaw's first touch anywhere on the object. Identical rule to
    the finger drop-depth mode, just with the whole boundary as the band."""
    if y_target is None:
        y_target = float(b.pts[:, 1].mean())
    all_idx = np.arange(len(b))
    la = anchor_in_band(b, all_idx, "left", reachable, y_target)
    ra = anchor_in_band(b, all_idx, "right", reachable, y_target)
    antip = -float(b.N[la] @ b.N[ra])
    antipodal = bool(antip >= np.cos(np.radians(40.0)))
    return GraspPair(la, ra, antipodal, la, ra, antip)
