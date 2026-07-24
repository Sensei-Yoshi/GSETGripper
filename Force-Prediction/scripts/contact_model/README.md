# contact_model — geometric contact-area estimation for the Fin-Ray gripper

Standalone mathematical testbed (deliberately **not** wired into the
Force-Prediction pipeline yet) that estimates the finger–object contact area
and contact fraction from a 2D object silhouette, for the compliant TPU
parallel-jaw Fin-Ray gripper.

## Pipeline

```
image ──extract_object_outline.py──▶ spline CSV (px)
      ──run_on_outline.py──▶ contact length / area / fraction + overlay
```

| file | role |
|---|---|
| `extract_object_outline.py` | rembg + periodic spline outline extraction (moved here from `scripts/`) |
| `contact_geometry.py` | Module 1 — mm scaling, CCW orientation, resampling, spline frames, signed curvature |
| `grasp_selection.py` | Module 2 — antipodal band pairing, pad window, first-touch anchor |
| `finger_drape.py` | Modules 3+4 — reachability (convex κ-test + rolling-disk closing test) and the draping walk with max-wrap arcs and δ-gap fringes |
| `contact_area.py` | Module 5 + orchestration — transverse width `w_eff`, area, contact fraction; public API `estimate_contact(...)` |
| `synthetic_shapes.py` | analytic shapes with closed-form contact truths |
| `viz.py` | Module 6 — overlay figure (contact, bridges, wrap arcs, κ profile) |
| `run_synthetic_tests.py` | accuracy harness + κ_max ranking-stability sweep |
| `run_on_outline.py` | run on a real extracted outline CSV |
| `capture_and_analyze.py` | **end-to-end**: camera SPACE-capture (or `--image`) → rembg + spline → contact model → one organized folder per object under `data/real_contact_area/` (raw photo, cutout, mask, spline overlay, CSV, SVG, contact figure, `summary.json` with all numbers incl. κ_max sweep) + a master `index.csv` |

## Algorithm (draping walk)

A point is **reachable** iff (1) κ ≤ κ_max (the pad cannot hug a bulge
tighter than its minimum bend radius r_min = 1/κ_max) and (2) the external
disk of radius r_min tangent there contains no other boundary point (the
rolling-ball / morphological-closing test — catches waists locally *and*
globally via KD-tree). The two tests are complementary: a rolling external
disk touches every point of a convex region regardless of curvature, so
closing alone cannot represent the convex failure case.

Per finger, the pad **anchors** at the extremal-x conformable point in the
window (the jaw's first touch; a true vertex protruding > `seat_tol` wins,
as on a pentagon corner), then the walk proceeds outward both ways:

- **contact** while reachable;
- on failure the finger **departs** on its max-wrap arc (radius r_min,
  tangent-continuous, curving toward the object);
- airborne gap `g = r_min − |b − c|` gives tolerance contact for `g ≤ δ`
  (the Hertz-like fringe falls out of the exact arc geometry) and a
  **re-landing** when the surface rises back through the arc (`g ≤ 0`).

Transverse width per contact point: `w_eff = min(w_pad, 2√(2 R_t δ))` with
the corrected axisymmetric transverse radius `R_t = r_parallel / |N_x|`
(Meusnier; exactly R everywhere on a sphere — the naive `r_parallel` alone
underestimates off-equator). Prismatic: `R_t = ∞ → w_eff = w_pad`.

**Contact fraction** (per finger) = area / (window_length × w_pad).

## Validation status

`run_synthetic_tests.py` — all 14 quantitative checks pass against
closed-form truths (worst error 1.8 mm on the waist, tol 2.5):

- gentle circle → full-window contact (exact)
- tight bulge (κ > κ_max) → δ-patch `2√(2δ/(1/R − κ_max))` (+0.09 mm)
- flat faces → face length + corner fringes (+0.01…0.26 mm)
- waist notch → bridged, rims contacted (−0.3…−1.8 mm)
- sphere area with corrected R_t (+0.7 %)
- noisy outline (σ = 0.05 mm) with smoothing → no false κ-trips (+0.12 mm)
- pentagon → face vs. vertex grasp; vertex collapses to a ~2.7 mm patch

κ_max sweep (r_min = 10/20/30 mm): well-separated shapes never change rank;
near-ties (square vs. waist, Δfraction < 0.03) can swap — that is the
sweep's job: report ranking gaps, not just orderings.

## Usage

```bash
VENV=/Users/premshah/Desktop/Robotics/GSET/env/bin/python
$VENV scripts/contact_model/run_synthetic_tests.py
$VENV scripts/contact_model/run_on_outline.py \
    data/MatForce/outline_outputs/plastic_cup_spline_points.csv \
    --px-per-mm 8.4 --object-type axisymmetric --k-max 2.0 --w-pad 12

# live end-to-end: SPACE = capture, click the edges of a 50 mm fiducial
$VENV scripts/contact_model/capture_and_analyze.py --ref-width-mm 50 \
    --object-type axisymmetric

# same pipeline on an existing photo
$VENV scripts/contact_model/capture_and_analyze.py \
    --image data/MatForce/plastic_cup.png --px-per-mm 8.0 --no-show
```

## Parameters

| name | default | meaning |
|---|---|---|
| `k_max` | 2.0 /mm | max conformable curvature — free parameter, sweep {1, 2, 4} |
| `delta` | 0.3 mm | gap tolerance ≈ TPU indentation at nominal grip force |
| `L` | 4.0 mm | pad contact length along the boundary |
| `w_pad` | from CAD | pad width (out of plane) |
| `ds` | 0.25 mm | resample step |
| `smoothing` | 0.2 mm | pre-differentiation smoothing for extracted outlines |

## Caveats (read before trusting numbers)

1. **px_per_mm is not optional.** Every mm parameter is meaningless without
   a fiducial-derived scale. Photograph a known-width reference in-scene.
2. **δ is the force proxy.** The whole model is geometry at one implied
   grip force; output is "projected contact at nominal force", and δ must
   scale if force does. At pad scale δ/r_min ≈ 0.6 is not small — sweep
   δ ∈ {0.1, 0.3} alongside κ_max.
3. **Bridged intervals use the optimistic max-wrap arc**, not beam
   mechanics; real Fin-Ray inversion can wrap more on convex objects
   (underestimate) and bow less into dips (overestimate). Relative
   rankings are the supported use, per the κ_max sweep.
4. The synthetic suite runs at structure scale (r_min ≈ 20 mm) purely for
   numerical resolution; the mathematics is scale-free.
