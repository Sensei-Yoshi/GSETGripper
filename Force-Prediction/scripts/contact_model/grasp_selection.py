"""Module 2 - grasp point / window selection.

Jaws close along +/-x. Candidate contact bands are boundary points whose
outward normal is within ``band_deg`` of the closing axis. Left/right
candidates are paired by height, scored by antipodality (N_l . N_r near -1)
with a mild pull toward ``y_target`` (default: centroid height).

Within the chosen pad window the physical first-touch point of an advancing
flat jaw is the extremal-x point, so the finger anchor is re-seated there
(ties broken toward the window centre). If no antipodal pair exists inside
the band (e.g. a regular pentagon: face opposes vertex) the selector falls
back to the global extremal-x points and flags the grasp non-antipodal.
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


def _anchor_in_window(
    b: Boundary,
    center: int,
    half_pts: int,
    side: str,
    reachable: np.ndarray,
    seat_tol: float = 1.0,
) -> int:
    """First-touch point of the advancing jaw within the pad window.

    The pad seats on the most extremal *conformable* point: spline overshoot
    at sharp corners can make an unreachable corner point the raw extremum
    by a hair, and anchoring there loses the adjacent flat. But if the raw
    extremum protrudes more than ``seat_tol`` beyond every reachable point
    (a true vertex, e.g. a pentagon corner), the jaw really does stop there.
    """
    idxs = (center + np.arange(-half_pts, half_pts + 1)) % len(b)
    sgn = -1.0 if side == "left" else 1.0
    proj = sgn * b.pts[idxs, 0]  # larger = closer to the advancing jaw
    ext = proj.max()

    reach_local = reachable[idxs]
    if reach_local.any():
        ext_r = proj[reach_local].max()
        if ext - ext_r <= seat_tol:
            ties = np.where(reach_local & (proj >= ext_r - 0.05))[0]
            pick = ties[np.argmin(np.abs(ties - half_pts))]
            return int(idxs[pick])

    ties = np.where(proj >= ext - 0.05)[0]
    pick = ties[np.argmin(np.abs(ties - half_pts))]
    return int(idxs[pick])


def select_grasp_pair(
    b: Boundary,
    L: float,
    reachable: np.ndarray,
    band_deg: float = 20.0,
    y_tol: float = 2.0,
    y_target: float | None = None,
    height_weight: float = 0.02,
) -> GraspPair:
    band = np.cos(np.radians(band_deg))
    nx = b.N[:, 0]
    y = b.pts[:, 1]
    if y_target is None:
        y_target = float(y.mean())

    stride = max(1, int(round(1.0 / b.ds)))  # ~1 mm candidate spacing
    left = np.where(nx <= -band)[0][::stride]
    right = np.where(nx >= band)[0][::stride]

    best: tuple[float, int, int] | None = None
    for li in left:
        dy = np.abs(y[right] - y[li])
        for ri in right[dy <= y_tol]:
            antip = -float(b.N[li] @ b.N[ri])
            score = antip - height_weight * (
                abs(y[li] - y_target) + abs(y[ri] - y_target)
            )
            if best is None or score > best[0]:
                best = (score, int(li), int(ri))

    if best is None:
        la = int(np.argmin(b.pts[:, 0]))
        ra = int(np.argmax(b.pts[:, 0]))
        return GraspPair(la, ra, False, la, ra, 0.0)

    score, lc, rc = best
    half_pts = min(int(round((L / 2) / b.ds)), len(b) // 2 - 1)
    return GraspPair(
        _anchor_in_window(b, lc, half_pts, "left", reachable),
        _anchor_in_window(b, rc, half_pts, "right", reachable),
        True,
        lc,
        rc,
        score,
    )
