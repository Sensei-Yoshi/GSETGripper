"""Module 1 - boundary preprocessing.

Conventions used across the contact_model package:

- Points are (x, y) in millimetres, y-up, forming one closed curve.
- The curve is stored counter-clockwise (CCW); ``build_boundary`` enforces this.
- Unit tangent T follows the CCW traversal; outward unit normal N = (T_y, -T_x).
- Signed curvature kappa > 0 on convex regions (bulging outward) and
  kappa < 0 on concave regions (dips/waists). Never take abs(): the two
  cases are handled differently downstream.

The smoothing knob is expressed in mm of allowed deviation (same convention
as extract_object_outline.py uses in pixels) so raw extracted splines can be
lightly re-smoothed before differentiating twice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import splev, splprep


@dataclass
class Boundary:
    """A closed CCW boundary resampled to ~uniform arc length."""

    pts: np.ndarray    # (M, 2) mm
    T: np.ndarray      # (M, 2) unit tangents
    N: np.ndarray      # (M, 2) unit outward normals
    kappa: np.ndarray  # (M,) signed curvature, 1/mm, >0 convex
    s: np.ndarray      # (M,) cumulative arc length, mm
    ds: float          # actual mean sample spacing, mm
    length: float      # total perimeter, mm

    def __len__(self) -> int:
        return len(self.pts)


def signed_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def ensure_ccw(pts: np.ndarray) -> np.ndarray:
    return pts if signed_area(pts) > 0 else pts[::-1].copy()


def polyline_length(pts: np.ndarray, closed: bool = True) -> float:
    p = np.vstack((pts, pts[:1])) if closed else pts
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def resample_closed(pts: np.ndarray, count: int) -> np.ndarray:
    """Sample a closed polyline at ``count`` equal arc-length intervals."""
    closed = np.vstack((pts, pts[:1]))
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    total = cum[-1]
    if total <= 0:
        raise ValueError("boundary has zero length")
    d = np.linspace(0.0, total, count, endpoint=False)
    return np.column_stack(
        (np.interp(d, cum, closed[:, 0]), np.interp(d, cum, closed[:, 1]))
    )


def build_boundary(
    pts_mm: np.ndarray, ds: float = 0.25, smoothing_mm: float = 0.0
) -> Boundary:
    """Fit a periodic spline to a closed point set and evaluate frames.

    Curvature comes from analytic spline derivatives (parametrisation
    invariant), not finite differences, so noise handling lives entirely
    in ``smoothing_mm``.
    """
    pts = ensure_ccw(np.asarray(pts_mm, dtype=float))
    rough_len = polyline_length(pts)
    n_fit = int(np.clip(round(rough_len / max(ds, 1e-3)), 128, 4000))
    uniform = resample_closed(pts, n_fit)
    closed = np.vstack((uniform, uniform[:1]))

    smoothing = len(closed) * smoothing_mm**2
    tck, _ = splprep(
        [closed[:, 0], closed[:, 1]],
        s=smoothing,
        per=True,
        k=min(3, len(closed) - 1),
    )

    u = np.linspace(0.0, 1.0, n_fit, endpoint=False)
    x, y = splev(u, tck)
    dx, dy = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)

    speed = np.hypot(dx, dy)
    speed = np.where(speed < 1e-12, 1e-12, speed)
    T = np.column_stack((dx / speed, dy / speed))
    N = np.column_stack((T[:, 1], -T[:, 0]))  # outward for CCW traversal
    kappa = (dx * ddy - dy * ddx) / speed**3

    out = np.column_stack((x, y))
    seg = np.linalg.norm(np.diff(np.vstack((out, out[:1])), axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg[:-1])))
    length = float(seg.sum())
    return Boundary(out, T, N, kappa, s, length / n_fit, length)
