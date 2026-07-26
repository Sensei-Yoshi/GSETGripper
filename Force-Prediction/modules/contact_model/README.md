# contact_model — projected two-pad contact fraction

Standalone geometric estimator for the TPU Fin-Ray parallel-jaw gripper. It
uses a calibrated 2D object silhouette to estimate the fraction of the two
4.2-inch pads covered by contiguous side contact:

```text
f_geometric = (left_contact_length + right_contact_length) / (2 * 106.68 mm)
f = max(minimum_contact_fraction, f_geometric)  for an antipodal grasp
```

The output is a dimensionless macroscopic geometry proxy. It is not physical
area in mm² and not microscopic Gecko adhesive contact area. Constant pad width
is assumed to be available wherever longitudinal contact exists, so width
cancels from the ratio.

Newly collected physical objects use this estimator as the authoritative
contact input for E1–E6. Historical synthetic fixtures are not reanalyzed.
For the stable Python fields, JSON schema, and examples for other code,
see [`docs/contact-fraction-integration.md`](../../docs/contact-fraction-integration.md).

## Pipeline

```text
RGB image → rembg mask → periodic outline spline → millimetre boundary
          → top-aligned pad window → side anchors → contiguous contact walk
          → combined fraction + diagnostic overlay
```

The CLI and Streamlit Contact Fraction tab both call `pipeline_core.analyze_image`,
so extraction, estimation, summaries, and v2 indexing share one code path.

## Physical rules

1. The pad window is `[object_top − 106.68 mm, object_top]`.
2. Left and right anchors are the outwardmost side-facing samples inside that
   window. Minor spline overshoot within 1 mm falls back to the reachable flat
   face; a genuinely protruding nonconformable feature still wins.
3. Both anchors must satisfy the 40° opposed-normal antipodal criterion.
4. A point is side-facing when
   `side_sign * N_x >= cos(side_angle_deg)`, with a 30° default.
5. A point is conformable when positive curvature is at most
   `1 / minimum_bend_radius` and the exterior rolling disk of that radius does
   not intersect another part of the boundary.
6. Contact walks outward from each anchor while pad material, side orientation,
   and conformability remain. The first failure ends the patch permanently;
   there is no idealized bridge or re-landing.
7. Actual boundary-segment lengths are integrated and capped at one pad length,
   preventing endpoint sampling from producing a fraction above one.
8. A valid antipodal grasp has a configurable minimum authoritative fraction
   of 0.05. This represents unresolved TPU seating contact; it does not invent
   green-path length. Non-antipodal grasps remain zero.

## Defaults

| Parameter | Default | Meaning |
|---|---:|---|
| `pad_length_mm` | 106.68 | active 4.2-inch pad length |
| `minimum_bend_radius_mm` | 20 | assembled-finger longitudinal bend limit |
| `side_angle_deg` | 30 | maximum normal deviation from jaw direction |
| `minimum_contact_fraction` | 0.05 | assumed seating-contact floor for an antipodal grasp |
| `ds` | 0.25 mm | boundary resampling target |
| `smoothing` | 0.2 mm | spline smoothing before curvature differentiation |
| radius sweep | 10, 20, 30 mm | sensitivity analysis; larger is more conservative |

`px_per_mm` remains mandatory and is valid only while the object-to-camera
geometry is unchanged.

## Outputs and persistence

Each new run writes the source image at the object root and the
cutout/mask/spline/contact artifacts plus schema-v2 `summary.json` under the
object's `contact_fraction/` directory. The primary result is
`combined_contact_fraction`; per-pad lengths and fractions remain diagnostics.
The summary also preserves `geometric_contact_fraction` and
`contact_floor_applied`, making the configured floor visible.

## Validation

```bash
../../env/bin/python -m pytest tests/test_contact_fraction.py
../../env/bin/python -m modules.contact_model.run_synthetic_tests
```

The pytest suite covers analytic rectangles/circles, curvature rejection,
antipodality, contiguity, resampling stability, and the committed water-bottle
and 3D-print outlines bundled under `modules/contact_model/test_contact_area/`.
Raw-image rembg coverage
is opt-in with `RUN_CONTACT_IMAGE_INTEGRATION=1` so normal tests stay offline.

## Usage

```bash
../../env/bin/python -m modules.contact_model.run_on_outline \
  modules/contact_model/test_contact_area/water_bottle/water_bottle_spline_points.csv \
  --px-per-mm 2.1852

../../env/bin/python -m modules.contact_model.capture_and_analyze \
  --dataset collected --image sample.png --px-per-mm 2.1852 --no-show
```
