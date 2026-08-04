"""Analytic test shapes in millimetres, y-up and approximately CCW."""

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
