# Contact-fraction integration reference

This document is the handoff contract for code that needs the gripper's
estimated surface-contact ratio. It covers the schema-v2 contact model under
`modules/contact_model/`, which supplies the contact value for newly collected
objects while historical fixture values remain unchanged.

## The short answer

The authoritative value is:

```python
ratio = estimate.combined_contact_fraction
```

It is a dimensionless `float` in `[0, 1]`. It is the fraction of the maximum
combined longitudinal contact of both 4.2-inch pads:

```text
combined_contact_fraction
    = max(0.05, geometric_contact_fraction) for an antipodal grasp
    = 0 otherwise

geometric_contact_fraction
    = (left_contact_length_mm + right_contact_length_mm)
      / (2 * 106.68 mm)
```

Constant pad width cancels. Do not interpret this value as square millimeters
or microscopic Gecko adhesive contact area.

When consuming a saved schema-v2 summary instead of a Python object, use:

```python
ratio = summary["results"]["combined_contact_fraction"]
feasible = summary["results"]["grasp_feasible"]
```

Do not use the legacy `mean_fraction`, `total_area_mm2`, `area_mm2`, or
`w_pad` fields. They belong to the retired area/wrapping model.

## Which entry point to use

There are three supported levels. Prefer the lowest level whose inputs you
already have.

### 1. An outline already expressed in millimeters

Use `contact_area.estimate_contact`. The input must be an `N x 2` NumPy array
representing one closed object outline in the gripper's longitudinal camera
plane.

- `x` is the jaw-closing axis;
- `y` points upward;
- coordinates are millimeters;
- clockwise or counter-clockwise input is accepted;
- the returned boundary is normalized to counter-clockwise order.

The contact model is part of the installed `modules` package:

```python
import numpy as np

from modules.contact_model import estimate_contact

points_mm: np.ndarray = ...  # shape (N, 2), x-closing and y-up
estimate = estimate_contact(
    points_mm,
    pad_length_mm=106.68,
    minimum_bend_radius_mm=20.0,
    side_angle_deg=30.0,
    minimum_contact_fraction=0.05,
    ds=0.25,
    smoothing_mm=0.2,
)

ratio = estimate.combined_contact_fraction
```

### 2. A saved `*_spline_points.csv` outline

Use `pipeline_core.outline_csv_to_mm` to convert image coordinates and scale,
then call `estimate_contact`:

```python
from modules.contact_model import estimate_contact, outline_csv_to_mm

points_mm = outline_csv_to_mm(
    csv_path,
    px_per_mm=2.1852,
    closing_axis="x",
)
estimate = estimate_contact(points_mm)
ratio = estimate.combined_contact_fraction
```

This path is deterministic and local. It is the preferred path for tests and
regressions because it does not require a camera or the `rembg` model.

### 3. A raw image requiring outline extraction

Use `pipeline_core.analyze_image`. It runs segmentation, outline fitting,
contact estimation, visualization, and schema-v2 summary generation:

```python
from modules.contact_model import ContactParams, analyze_image

params = ContactParams(
    px_per_mm=2.1852,
    pad_length_mm=106.68,
    minimum_bend_radius_mm=20.0,
    side_angle_deg=30.0,
    minimum_contact_fraction=0.05,
    sweep_radii_mm=(10.0, 20.0, 30.0),
)

estimate, summary, paths = analyze_image(
    image_path=image_path,
    run_dir=run_dir,
    name="object_name",
    params=params,
)

ratio_from_object = estimate.combined_contact_fraction
ratio_from_summary = summary["results"]["combined_contact_fraction"]
```

This path requires a valid `px_per_mm` calibration at the same camera-to-object
distance used for the image. Normal unit tests should use saved spline CSVs
instead.

## Returned Python contract

`estimate_contact` returns a `ContactEstimate`. The fields most likely to be
needed by other code are:

| Field | Type | Meaning |
|---|---|---|
| `combined_contact_fraction` | `float` | Authoritative two-pad ratio in `[0, 1]` |
| `combined_contact_length` | `float` | Sum of both valid contact lengths, capped at `213.36 mm` |
| `geometric_contact_fraction` | `float` | Green-path length ratio before the minimum-contact floor |
| `contact_floor_applied` | `bool` | Whether the authoritative ratio was raised to the configured floor |
| `minimum_contact_fraction` | `float` | Configured floor for a valid antipodal grasp |
| `feasible` | `bool` | Whether a valid antipodal grasp has geometric or assumed minimum contact |
| `pair.antipodal` | `bool` | Whether the two red anchors pass the 40° opposed-normal test |
| `left.contact_length` | `float` | Valid contiguous left-pad length in millimeters |
| `right.contact_length` | `float` | Valid contiguous right-pad length in millimeters |
| `left.fraction` | `float` | `left.contact_length / pad_length` |
| `right.fraction` | `float` | `right.contact_length / pad_length` |
| `left.contact_points` | `ndarray` | Ordered points drawn as the left green path |
| `right.contact_points` | `ndarray` | Ordered points drawn as the right green path |
| `pad_band` | `(float, float)` | Bottom and top of the active pad window in millimeters |

For downstream decision-making, retain both the ratio and feasibility flag:

```python
contact = {
    "combined_contact_fraction": estimate.combined_contact_fraction,
    "geometric_contact_fraction": estimate.geometric_contact_fraction,
    "contact_floor_applied": estimate.contact_floor_applied,
    "grasp_feasible": estimate.feasible,
    "antipodal_grasp": estimate.pair.antipodal,
}
```

The authoritative fraction is already zero for a rejected non-antipodal
grasp. The flags explain why a zero occurred and should not be discarded from
research artifacts.

The minimum-contact floor is an explicit physical assumption about unavoidable
TPU seating/contact that the macroscopic outline walk cannot resolve. It does
not create green-path length. Keep `geometric_contact_fraction` when the
difference matters scientifically, and calibrate the `0.05` default against
physical measurements when those data become available.

## Schema-v2 JSON and CSV contract

Every new image-pipeline run writes a `summary.json` with:

```json
{
  "schema_version": 2,
  "metric": "projected_two_pad_contact_fraction",
  "params": {
    "minimum_contact_fraction": 0.05
  },
  "results": {
    "grasp_feasible": true,
    "antipodal_grasp": true,
    "left": {
      "contact_length_mm": 94.5,
      "pad_length_mm": 106.68,
      "contact_fraction": 0.8858
    },
    "right": {
      "contact_length_mm": 95.459,
      "pad_length_mm": 106.68,
      "contact_fraction": 0.8948
    },
    "combined_contact_length_mm": 189.959,
    "geometric_contact_fraction": 0.8903,
    "contact_floor_applied": false,
    "combined_contact_fraction": 0.8903
  }
}
```

Each object stores its authoritative result in
`objects/<object_id>/contact_fraction/summary.json`. The Python
`ContactEstimate` retains full floating-point precision; saved summary contact
fractions are rounded to four decimal places.

## Mathematical model

Let the resampled object boundary be `p(s) = (x(s), y(s))`, with outward unit
normal `N(s)` and signed curvature `kappa(s)`. The model applies these steps:

1. **Pad window.** Only points with
   `object_top - L <= y <= object_top` are considered, where `L = 106.68 mm`.
2. **Side orientation.** For the left pad `side_sign = -1`; for the right pad
   `side_sign = +1`. A point is side-facing when
   `side_sign * N_x >= cos(30°)`. This excludes horizontal top and bottom
   surfaces.
3. **Anchors.** The red anchor on each side is the outwardmost side-facing
   point in the pad window. Flat-face ties resolve toward the window center.
4. **Antipodality.** With anchor normals `N_L` and `N_R`, the pair passes when
   `-N_L dot N_R >= cos(40°)`. A failure makes the grasp infeasible and sets
   the authoritative fraction to zero.
5. **Bend limit.** With minimum bend radius `R_min`, convex boundary curvature
   must satisfy `kappa <= 1 / R_min`. The default `R_min = 20 mm` corresponds
   to `k_max = 0.05 1/mm`.
6. **Concave accessibility.** An exterior disk of radius `R_min`, tangent at a
   candidate boundary point, must not contain another boundary point. This
   rejects pockets the finite-radius pad cannot enter.
7. **Contiguous walk.** Each pad walks away from its anchor in both boundary
   directions while material budget, window, angle, bend, and accessibility
   constraints pass. The first failure ends that direction permanently; the
   model never bridges a gap or re-lands later.
8. **Integration.** Accepted Euclidean boundary-segment lengths are summed.
   A last partial segment is interpolated when the pad budget ends inside it.
9. **Minimum physical contact.** For an accepted antipodal grasp, the
   authoritative fraction is no smaller than `minimum_contact_fraction`. This
   represents unresolved finite seating contact and does not alter the green
   geometric paths. A rejected non-antipodal grasp remains zero.

The final result is:

```text
ell_left  = min(valid left boundary length, L)
ell_right = min(valid right boundary length, L)

f_geometric = clip((ell_left + ell_right) / (2L), 0, 1)

f = max(minimum_contact_fraction, f_geometric)  if antipodal
f = 0                                           otherwise
```

## Defaults and invariants

| Quantity | Default or invariant |
|---|---:|
| Pad length `L` | `106.68 mm` (4.2 inches) |
| Minimum bend radius | `20 mm` |
| Bend-radius validation sweep | `10, 20, 30 mm` |
| Side-normal tolerance | `30°` |
| Antipodal tolerance | `40°` |
| Minimum authoritative contact fraction | `0.05` |
| Resampling `ds` | `0.25 mm` |
| Per-pad contact length | `[0, 106.68] mm` |
| Combined contact length | `[0, 213.36] mm` |
| Geometric contact fraction | `[0, 1]` |
| Authoritative fraction, antipodal | `[0.05, 1]` with default config |
| Authoritative fraction, non-antipodal | `0` |

Increasing `minimum_bend_radius_mm` is more conservative. The saved-fixture
regressions require `f(10 mm) >= f(20 mm) >= f(30 mm)`.

## Relationship to the force-prediction pipeline

The active E1–E4 contract retains the field name
`projected_contact_fraction` for compatibility. Newly collected physical
objects populate it from schema-v2 `combined_contact_fraction` and preserve
model provenance in `Meta`. ExpForce remains a synthetic fixture and is not
reanalyzed or presented as physical contact evidence.

## Change and test checklist

When modifying the contact estimator or consuming its output:

- keep `combined_contact_fraction` as the authoritative field name;
- preserve dimensionless `[0, 1]` semantics and the two-pad denominator;
- keep the floor explicit and preserve the pre-floor geometric fraction;
- store `grasp_feasible` and `antipodal_grasp` beside the ratio;
- never restore pad-width or square-millimeter calculations;
- never bridge invalid outline regions;
- preserve one authoritative schema-v2 summary inside each object folder;
- run `pytest tests/test_contact_fraction.py`;
- inspect the water-bottle and 3D-print v2 overlays after geometry changes.

The implementation details live in `modules/contact_model/contact_area.py`,
`grasp_selection.py`, `finger_drape.py`, and `pipeline_core.py`. The model's
research-facing overview is in `modules/contact_model/README.md`.
