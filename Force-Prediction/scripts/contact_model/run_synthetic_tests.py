"""Accuracy harness: run the contact model on analytic shapes and compare
against closed-form ground truths, then sweep k_max for ranking stability.

    /Users/premshah/Desktop/Robotics/GSET/env/bin/python \
        scripts/contact_model/run_synthetic_tests.py

Figures land in scripts/contact_model/test_outputs/. Exit code 1 on any
quantitative FAIL.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contact_area import FingerGeometry, estimate_contact
from synthetic_shapes import (
    add_noise, circle, pentagon, rounded_rect,
    truth_circle, truth_square, truth_waist,
)
from viz import plot_estimate

OUT = Path(__file__).resolve().parent / "test_outputs"

# Structure-scale test parameters (mathematics is scale-free; see README).
K_MAX = 0.05    # 1/mm  -> r_min = 20 mm
DELTA = 0.3     # mm
L = 40.0        # mm window
W_PAD = 12.0    # mm
DS = 0.25       # mm

CASES = [
    dict(
        name="gentle_circle_R40", pts=circle(40.0), smoothing=0.0,
        object_type="prismatic", metric="length", tol=0.8,
        truth=lambda k: truth_circle(40.0, k, DELTA, L),
    ),
    dict(
        name="tight_circle_R8", pts=circle(8.0), smoothing=0.0,
        object_type="prismatic", metric="length", tol=0.8,
        truth=lambda k: truth_circle(8.0, k, DELTA, L),
    ),
    dict(
        name="square60_rc6", pts=rounded_rect(60.0, 60.0, 6.0), smoothing=0.0,
        object_type="prismatic", metric="length", tol=0.8,
        truth=lambda k: truth_square(60.0, 6.0, k, DELTA, L),
    ),
    dict(
        name="square30_rc4", pts=rounded_rect(30.0, 30.0, 4.0), smoothing=0.0,
        object_type="prismatic", metric="length", tol=1.5,
        truth=lambda k: truth_square(30.0, 4.0, k, DELTA, L),
    ),
    dict(
        name="waist_notch5", pts=rounded_rect(50.0, 90.0, 8.0, notch_r=5.0),
        smoothing=0.0, object_type="prismatic", metric="length", tol=2.5,
        truth=lambda k: truth_waist(5.0, k, DELTA, L),
    ),
    dict(
        name="sphere_R40_area", pts=circle(40.0), smoothing=0.0,
        object_type="axisymmetric", metric="area", tol=12.0,
        truth=lambda k: truth_circle(40.0, k, DELTA, L)
        * min(W_PAD, 2.0 * np.sqrt(2.0 * 40.0 * DELTA)),
    ),
    dict(
        name="noisy_circle_R40", pts=add_noise(circle(40.0), 0.05, seed=0),
        smoothing=0.25, object_type="prismatic", metric="length", tol=2.0,
        truth=lambda k: truth_circle(40.0, k, DELTA, L),
    ),
    dict(
        name="pentagon_s50", pts=pentagon(50.0), smoothing=0.1,
        object_type="prismatic", metric=None, tol=None, truth=None,
    ),
]

# ---------------------------------------------------------------------------
# Finger drop-depth cases: circle = "orange" of radius 30 resting on the
# table (centre at y=30, top at 60). Truths per finger, k_max=0.05.
# ---------------------------------------------------------------------------

ORANGE = circle(30.0) + np.array([0.0, 30.0])
BALL = circle(8.0) + np.array([0.0, 8.0])
TALL = rounded_rect(50.0, 150.0, 8.0, notch_r=5.0) + np.array([0.0, 75.0])

FINGER_CASES = [
    dict(
        # long finger, pad reaches the equator: full-pad conformal contact
        name="F1_orange_pad_at_equator",
        pts=ORANGE,
        finger=FingerGeometry(finger_length=100.0, pad_length=40.0),
        truth=40.0, tol=0.8, feasible=True,
    ),
    dict(
        # the user's issue: pad band sits ABOVE the equator (band 32..62 vs
        # equator 30) -> anchor at band bottom, zero pad below it, contact
        # limited to the 30 mm of pad above -> 30, not the free-model 40
        name="F2_orange_pad_above_equator",
        pts=ORANGE,
        finger=FingerGeometry(
            finger_length=100.0, pad_length=30.0, pad_start=30.0
        ),
        truth=30.0, tol=1.0, feasible=True,
    ),
    dict(
        # object top (16) entirely below the pad's lower edge (22): the
        # grasp is infeasible and must report zero, not invent contact
        name="F3_ball_below_pad",
        pts=BALL,
        finger=FingerGeometry(
            finger_length=100.0, pad_length=30.0, pad_start=20.0
        ),
        truth=0.0, tol=0.01, feasible=False,
    ),
    dict(
        # tall object: palm hits the top first -> h_tip = 150+5-80 = 75,
        # band 75..115 grasps the upper region and clips the notch top:
        # 20 mm up-wall + 15 mm down-wall + corner fringe past the notch rim
        name="F4_tall_waist_top_grasp",
        pts=TALL,
        finger=FingerGeometry(
            finger_length=80.0, pad_length=40.0
        ),
        truth=35.5, tol=2.5, feasible=True,
    ),
]

SWEEP_K = [1.0 / 30.0, 0.05, 0.10]
SWEEP_CASES = ["gentle_circle_R40", "tight_circle_R8", "square30_rc4", "waist_notch5"]


def run_case(case: dict, k_max: float):
    return estimate_contact(
        case["pts"], k_max=k_max, delta=DELTA, L=L, w_pad=W_PAD, ds=DS,
        smoothing_mm=case["smoothing"], object_type=case["object_type"],
    )


def main() -> int:
    rows = []
    failures = 0
    print(f"contact_model synthetic suite  "
          f"(k_max={K_MAX}/mm, r_min={1/K_MAX:.0f} mm, delta={DELTA} mm, "
          f"L={L} mm, ds={DS} mm)\n")

    header = (f"{'case':24s} {'finger':6s} {'est':>8s} {'truth':>8s} "
              f"{'err':>7s} {'tol':>5s}  status")
    print(header)
    print("-" * len(header))

    for case in CASES:
        est = run_case(case, K_MAX)
        plot_estimate(est, OUT / f"{case['name']}.png",
                      f"{case['name']}  (k_max={K_MAX}/mm)")

        if case["metric"] is None:
            print(f"{case['name']:24s} {'-':6s} "
                  f"{est.left.contact_length:8.2f} {'behav.':>8s} "
                  f"{'-':>7s} {'-':>5s}  REPORT "
                  f"(antipodal={est.pair.antipodal}, "
                  f"R contact={est.right.contact_length:.2f} mm)")
            continue

        truth = float(case["truth"](K_MAX))
        for name, f in (("L", est.left), ("R", est.right)):
            if case["metric"] == "length":
                got = f.contact_length
            else:
                got = f.area
            err = got - truth
            ok = abs(err) <= case["tol"]
            failures += 0 if ok else 1
            unit = "mm" if case["metric"] == "length" else "mm2"
            rows.append((case["name"], name, got, truth, err, ok))
            print(f"{case['name']:24s} {name:6s} {got:8.2f} {truth:8.2f} "
                  f"{err:+7.2f} {case['tol']:5.1f}  "
                  f"{'PASS' if ok else 'FAIL'} ({unit})")

    # ------------------------------------------------------------------
    # finger drop-depth cases
    # ------------------------------------------------------------------
    print("\nfinger drop-depth cases (pad placement constrained by geometry):")
    for case in FINGER_CASES:
        est = estimate_contact(
            case["pts"], k_max=K_MAX, delta=DELTA, w_pad=W_PAD, ds=DS,
            smoothing_mm=0.0, object_type="prismatic", finger=case["finger"],
        )
        plot_estimate(est, OUT / f"{case['name']}.png",
                      f"{case['name']}  (k_max={K_MAX}/mm)")
        feas_ok = est.feasible == case["feasible"]
        for name, f in (("L", est.left), ("R", est.right)):
            err = f.contact_length - case["truth"]
            ok = abs(err) <= case["tol"] and feas_ok
            failures += 0 if ok else 1
            print(f"{case['name']:28s} {name:3s} {f.contact_length:8.2f} "
                  f"{case['truth']:8.2f} {err:+7.2f} {case['tol']:5.1f}  "
                  f"{'PASS' if ok else 'FAIL'} "
                  f"(feasible={est.feasible})")

    # ------------------------------------------------------------------
    # k_max sweep: is the ranking of shapes by contact fraction stable?
    # ------------------------------------------------------------------
    print("\nk_max sweep - 1D contact fraction (contact_length / window):")
    print(f"{'k_max':>7s} {'r_min':>6s}  " +
          "  ".join(f"{n[:14]:>14s}" for n in SWEEP_CASES) + "   ranking")
    orders = []
    for k in SWEEP_K:
        fracs = {}
        for name in SWEEP_CASES:
            case = next(c for c in CASES if c["name"] == name)
            est = run_case(case, k)
            fracs[name] = est.left.contact_length / est.left.window_length
        order = sorted(fracs, key=fracs.get, reverse=True)
        orders.append(order)
        print(f"{k:7.3f} {1/k:6.1f}  " +
              "  ".join(f"{fracs[n]:14.3f}" for n in SWEEP_CASES) +
              "   " + " > ".join(o.split('_')[0] for o in order))
    stable = all(o == orders[0] for o in orders)
    print(f"\nranking stable across k_max sweep: {stable}")

    print(f"\nfigures: {OUT}")
    print(f"{failures} quantitative failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
