"""Module 5 + orchestration - transverse width, area, contact fraction,
and the finger drop-depth model.

Transverse (out-of-plane) effective width per contact point:

    w_eff = min(w_pad, 2 * sqrt(2 * R_t * delta))

with the *corrected* transverse radius for axisymmetric objects:

    R_t = r_parallel / |N_x|

where r_parallel is the horizontal distance from the symmetry axis and N_x
the horizontal component of the outward normal (Meusnier). For a sphere this
is exactly R everywhere; the naive r_parallel alone underestimates R_t away
from the equator. Prismatic objects use R_t = inf, i.e. w_eff = w_pad.

Drop-depth model (``FingerGeometry``): the gripper descends from above, so
the pad's active band is NOT freely placeable. With the object resting on
the table (table = min y of the silhouette):

    h_tip   = table + max(tip_clearance,
                          object_height + palm_standoff - finger_length)
    pad band = [h_tip + pad_start, h_tip + pad_start + pad_length]

Anchors are restricted to the band, walk budgets are the pad material above
and below the anchor (asymmetric), the walk clamps at the fingertip height,
and a band entirely above the object top is an INFEASIBLE grasp (zero
contact) rather than an invented one.

Top-anchored placement (``pad_top_anchored=True``, the application default):
the pad hangs from the object's highest point, so the contact band is the top
``L`` of the object ([y_top - L, y_top]) and everything below is disregarded
- the finger approaches from above and cannot reach or wrap under the base.
``finger=None`` and ``pad_top_anchored=False`` keeps the free-placement
behaviour (pad centred wherever antipodality is best).

Contact fraction (per finger) = area / (pad_length * w_pad).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from contact_geometry import Boundary, build_boundary
from finger_drape import WrapArc, drape, reachability_mask
from grasp_selection import GraspPair, anchor_in_band, select_grasp_pair


@dataclass
class FingerGeometry:
    """Vertical-finger geometry for the drop-depth model (all mm, from CAD)."""

    finger_length: float        # palm mount to fingertip
    pad_length: float           # active pad extent along the finger
    pad_start: float = 0.0      # fingertip to the pad's lower edge
    tip_clearance: float = 2.0  # minimum fingertip height above the table
    palm_standoff: float = 5.0  # palm clearance above the object top


@dataclass
class FingerResult:
    side: str
    anchor: int                  # -1 when the grasp is infeasible
    window_idx: np.ndarray       # boundary indices, ordered along the window
    contact: np.ndarray          # bool, aligned with window_idx
    gap: np.ndarray              # mm, aligned with window_idx
    w_eff: np.ndarray            # mm, aligned with window_idx
    arcs: list[WrapArc]
    contact_length: float        # mm
    window_length: float         # mm (pad length in finger mode)
    area: float                  # mm^2
    fraction: float              # dimensionless


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
    finger: FingerGeometry | None = None
    pad_band: tuple[float, float] | None = None  # (y_lo, y_hi), abs mm
    tip_height: float | None = None              # abs mm
    feasible: bool = True

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


def _empty_finger(side: str, pad_length: float) -> FingerResult:
    z = np.zeros(0)
    return FingerResult(
        side, -1, np.zeros(0, dtype=int), np.zeros(0, dtype=bool), z, z,
        [], 0.0, pad_length, 0.0, 0.0,
    )


def _steps_by_direction(
    b: Boundary, anchor: int, up_len: float, down_len: float
) -> dict[int, int]:
    """Map walk directions (+1/-1 along the boundary) to pad-material
    budgets above/below the anchor. Which direction is 'up' depends on
    which side of the CCW curve the anchor sits on."""
    n = len(b)
    cap = n // 2 - 1
    probe = b.pts[(anchor + 3) % n, 1] - b.pts[anchor, 1]
    s_up = min(int(round(up_len / b.ds)), cap)
    s_dn = min(int(round(down_len / b.ds)), cap)
    if probe >= 0:
        return {+1: s_up, -1: s_dn}
    return {+1: s_dn, -1: s_up}


def _analyze_finger(
    b: Boundary,
    anchor: int,
    side: str,
    steps: dict[int, int],
    k_max: float,
    delta: float,
    reachable: np.ndarray,
    w_pad: float,
    object_type: str,
    axis_x: float,
    denom_length: float,
    y_min: float | None,
) -> FingerResult:
    side_sign = -1.0 if side == "left" else 1.0
    record: dict[int, tuple[bool, float]] = {}
    arcs: list[WrapArc] = []
    for direction in (+1, -1):
        rec, a = drape(
            b, anchor, direction, steps[direction], k_max, delta,
            reachable, side_sign, y_min,
        )
        for idx, (c, g) in rec.items():
            if idx in record:
                pc, pg = record[idx]
                record[idx] = (pc or c, min(pg, g))
            else:
                record[idx] = (c, g)
        arcs.extend(a)

    window_idx = (anchor + np.arange(-steps[-1], steps[+1] + 1)) % len(b)
    contact = np.array([record.get(i, (False, np.inf))[0] for i in window_idx])
    gap = np.array([record.get(i, (False, np.inf))[1] for i in window_idx])
    w_eff = transverse_width(b, window_idx, object_type, delta, w_pad, axis_x)

    contact_length = float(contact.sum() * b.ds)
    area = float(np.sum(w_eff[contact]) * b.ds)
    fraction = min(area / (denom_length * w_pad), 1.0)
    return FingerResult(
        side, anchor, window_idx, contact, gap, w_eff, arcs,
        contact_length, denom_length, area, fraction,
    )


def _analyze_band(
    b: Boundary,
    reachable: np.ndarray,
    pad_lo: float,
    pad_hi: float,
    pad_L: float,
    k_max: float,
    delta: float,
    w_pad: float,
    object_type: str,
    axis_x: float,
    y_min: float,
) -> tuple[GraspPair, FingerResult, FingerResult, bool]:
    """Grasp + drape restricted to the vertical pad band [pad_lo, pad_hi].

    Anchors are chosen inside the band; each finger's walk is budgeted by the
    pad material above/below its anchor and clamped at ``y_min`` so contact
    never runs below the band (e.g. it cannot wrap under the object base).
    """
    y = b.pts[:, 1]
    band_idx = np.where((y >= pad_lo) & (y <= pad_hi))[0]
    if len(band_idx) == 0:
        return (
            GraspPair(-1, -1, False, -1, -1, 0.0),
            _empty_finger("left", pad_L), _empty_finger("right", pad_L), False,
        )
    pad_center = 0.5 * (pad_lo + pad_hi)
    la = anchor_in_band(b, band_idx, "left", reachable, pad_center)
    ra = anchor_in_band(b, band_idx, "right", reachable, pad_center)
    antip = -float(b.N[la] @ b.N[ra])
    pair = GraspPair(
        la, ra, bool(antip >= np.cos(np.radians(40.0))), la, ra, antip
    )
    out = []
    for anchor, side in ((la, "left"), (ra, "right")):
        y_a = float(b.pts[anchor, 1])
        steps = _steps_by_direction(b, anchor, pad_hi - y_a, y_a - pad_lo)
        out.append(_analyze_finger(
            b, anchor, side, steps, k_max, delta, reachable,
            w_pad, object_type, axis_x, pad_L, y_min,
        ))
    return pair, out[0], out[1], True


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
    finger: FingerGeometry | None = None,
    pad_top_anchored: bool = False,
) -> ContactEstimate:
    """End-to-end estimate: boundary -> grasp -> drape -> area.

    Placement precedence:
      1. ``finger`` set  -> drop-depth model (pad band from finger geometry;
         ``finger.pad_length`` supersedes ``L``).
      2. ``pad_top_anchored`` -> the pad hangs from the object's highest point:
         the contact band is the top ``L`` of the object ([y_top - L, y_top])
         and everything below is disregarded, since the finger approaches from
         above and cannot reach or wrap under the base.
      3. otherwise -> free placement (pad centred on the best antipodal band).
    """
    b = build_boundary(pts_mm, ds=ds, smoothing_mm=smoothing_mm)
    reachable = reachability_mask(b, k_max)
    axis_x = float(b.pts[:, 0].mean())
    y_top = float(b.pts[:, 1].max())

    # 1) finger drop-depth model
    if finger is not None:
        table_y = float(b.pts[:, 1].min())
        h_tip = table_y + max(
            finger.tip_clearance,
            (y_top - table_y) + finger.palm_standoff - finger.finger_length,
        )
        pad_lo = h_tip + finger.pad_start
        pad_hi = pad_lo + finger.pad_length
        pair, left, right, feasible = _analyze_band(
            b, reachable, pad_lo, pad_hi, finger.pad_length, k_max, delta,
            w_pad, object_type, axis_x, h_tip,
        )
        return ContactEstimate(
            b, reachable, pair, left, right, k_max, delta, finger.pad_length,
            w_pad, object_type, finger, (pad_lo, pad_hi), h_tip, feasible,
        )

    # 2) top-anchored pad: contact band = top L of the object
    if pad_top_anchored:
        pad_hi = y_top
        pad_lo = y_top - L
        pair, left, right, feasible = _analyze_band(
            b, reachable, pad_lo, pad_hi, L, k_max, delta, w_pad,
            object_type, axis_x, pad_lo,
        )
        return ContactEstimate(
            b, reachable, pair, left, right, k_max, delta, L, w_pad,
            object_type, None, (pad_lo, pad_hi), pad_lo, feasible,
        )

    # 3) free placement (low-level default; CLI/synthetic tests)
    pair = select_grasp_pair(b, reachable, y_target=y_target)
    half_pts = min(int(round((L / 2) / b.ds)), len(b) // 2 - 1)
    steps = {+1: half_pts, -1: half_pts}
    denom = (2 * half_pts + 1) * b.ds
    left = _analyze_finger(
        b, pair.left_anchor, "left", steps, k_max, delta, reachable,
        w_pad, object_type, axis_x, denom, None,
    )
    right = _analyze_finger(
        b, pair.right_anchor, "right", steps, k_max, delta, reachable,
        w_pad, object_type, axis_x, denom, None,
    )
    return ContactEstimate(
        b, reachable, pair, left, right, k_max, delta, L, w_pad, object_type,
    )
