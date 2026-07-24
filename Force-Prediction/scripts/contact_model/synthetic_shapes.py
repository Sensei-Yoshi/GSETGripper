"""Analytic test shapes (mm, y-up, ~CCW) with closed-form contact truths.

The suite runs at "structure scale" (r_min ~ 10-30 mm) purely for numerical
resolution; the mathematics is scale-free, so validation here transfers
directly to the pad-scale defaults (k_max ~ 1-4 /mm) used on real outlines.
"""

from __future__ import annotations

import numpy as np

_STEP = 0.1  # polyline densification step, mm


def _line(p0, p1, step=_STEP):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / step)))
    t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
    return p0 + t * (p1 - p0)


def _arc(c, r, a0, a1, step=_STEP):
    c = np.asarray(c, float)
    n = max(2, int(np.ceil(r * abs(a1 - a0) / step)))
    a = np.linspace(a0, a1, n, endpoint=False)
    return c + r * np.column_stack((np.cos(a), np.sin(a)))


def circle(R: float) -> np.ndarray:
    return _arc((0.0, 0.0), R, 0.0, 2.0 * np.pi)


def rounded_rect(
    w: float, h: float, rc: float, notch_r: float | None = None
) -> np.ndarray:
    """CCW rounded rectangle, optionally with semicircular waist notches
    carved into the left and right sides at mid-height (bottle-shoulder
    analogue; the notch joins the wall at 90-degree corners)."""
    w2, h2 = w / 2.0, h / 2.0
    segs = []
    if notch_r:
        segs.append(_line((w2, -(h2 - rc)), (w2, -notch_r)))
        a = np.linspace(-np.pi / 2, np.pi / 2, max(2, int(np.pi * notch_r / _STEP)), endpoint=False)
        segs.append(np.column_stack((w2 - notch_r * np.cos(a), notch_r * np.sin(a))))
        segs.append(_line((w2, notch_r), (w2, h2 - rc)))
    else:
        segs.append(_line((w2, -(h2 - rc)), (w2, h2 - rc)))
    segs.append(_arc((w2 - rc, h2 - rc), rc, 0.0, np.pi / 2))
    segs.append(_line((w2 - rc, h2), (-(w2 - rc), h2)))
    segs.append(_arc((-(w2 - rc), h2 - rc), rc, np.pi / 2, np.pi))
    if notch_r:
        segs.append(_line((-w2, h2 - rc), (-w2, notch_r)))
        a = np.linspace(np.pi / 2, -np.pi / 2, max(2, int(np.pi * notch_r / _STEP)), endpoint=False)
        segs.append(np.column_stack((-w2 + notch_r * np.cos(a), notch_r * np.sin(a))))
        segs.append(_line((-w2, -notch_r), (-w2, -(h2 - rc))))
    else:
        segs.append(_line((-w2, h2 - rc), (-w2, -(h2 - rc))))
    segs.append(_arc((-(w2 - rc), -(h2 - rc)), rc, np.pi, 3 * np.pi / 2))
    segs.append(_line((-(w2 - rc), -h2), (w2 - rc, -h2)))
    segs.append(_arc((w2 - rc, -(h2 - rc)), rc, 3 * np.pi / 2, 2 * np.pi))
    return np.vstack(segs)


def pentagon(side: float) -> np.ndarray:
    """Regular pentagon with one flat face on the left and a vertex on the
    right - the classic non-antipodal parallel-jaw case."""
    rc = side / (2.0 * np.sin(np.pi / 5.0))
    angles = np.radians([0, 72, 144, 216, 288])
    verts = rc * np.column_stack((np.cos(angles), np.sin(angles)))
    segs = [
        _line(verts[i], verts[(i + 1) % 5]) for i in range(5)
    ]
    return np.vstack(segs)


def add_noise(pts: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return pts + rng.normal(0.0, sigma, pts.shape)


# ---------------------------------------------------------------------------
# Closed-form per-finger contact-length truths (quadratic small-gap forms)
# ---------------------------------------------------------------------------


def truth_circle(R: float, k_max: float, delta: float, L: float) -> float:
    if 1.0 / R <= k_max:
        return min(L, np.pi * R)  # fully conformable inside the window
    dk = 1.0 / R - k_max
    return min(2.0 * np.sqrt(2.0 * delta / dk), L)


def truth_square(
    side: float, rc: float, k_max: float, delta: float, L: float
) -> float:
    flat = side - 2.0 * rc
    if flat >= L:
        return L
    dk = 1.0 / rc - k_max
    fringe = np.sqrt(2.0 * delta / dk) if dk > 0 else np.pi * rc / 2.0
    return flat + 2.0 * fringe


def truth_waist(
    notch_r: float, k_max: float, delta: float, L: float
) -> float:
    if k_max >= 1.0 / notch_r:
        return L  # dip fully reachable
    # semicircular notch joins the wall at 90-degree corners, so the
    # delta-fringe past each corner is O(delta), not the tangent-continuous
    # quadratic fringe; ~1 mm covers both fringes plus re-land chatter.
    return L - np.pi * notch_r + 1.0
