# Ground-Truth Data Collection — Standard Operating Procedure

Goal: for every object, measure the minimum stationary-finger normal force
`F*(o, g)` that lifts it, for **both** gecko and silicone, under one fixed
protocol. Implemented by `force_prediction/collect.py`.

## Force convention (non-negotiable)
`F*` is the normal force read by the load cell **behind the stationary finger**,
in newtons. Never doubled, never summed across fingers. Same convention in the
dataset, physics, prompts, predictions, and metrics.

## Per object-gripper procedure
1. Mount the TPU–gecko or TPU–silicone finger; clean the pad (documented cleaner).
2. Place the object at the fixed, centered grasp location.
3. `close_until_contact()`.
4. For gecko, apply the fixed seating/shear displacement (log displacement + dwell).
5. Attempt the standardized lift (fixed speed, height, hold time).
6. On failure: fully **release and reset** (gecko has load-history effects).
7. Step the normal force according to the configured search procedure; repeat
   until lift or safe limit.
8. Repeat the whole minimum-force measurement the configured number of times;
   store the **median**.
9. If it never lifts within the safe limit → record `feasible=false`,
   `failed_at_limit_n = limit` (never store the limit as the minimum).

### Coarse-to-fine search
`collect.py` currently brackets the minimum in **1.0 N** steps, then refines in
**0.01 N** steps within the bracketing interval. These values are configured
under `collection` in `config.yaml`. They describe measurement resolution only:
the controller and all model predictions remain continuous from `0` to `8 N`.

## Hold constant (and document)
Finger geometry, pad area, object orientation, grasp height, closing speed, force
increment, gecko seating displacement + dwell, lift speed, lift height, required
hold duration, pad cleaning procedure, load-cell calibration.

## Randomize / log
- **Counterbalance** gecko vs silicone trial order so pad aging / timing do not
  correlate with one material.
- Log per session: temperature, humidity, pad id. (Pad-wear tracking is a
  backlog item — see `docs/backlog.md`.)

## Coverage targets
Spread objects across all five roughness classes, broad/narrow contact,
light/heavy mass, porous/nonporous, rigid/deformable — and **intentionally
include borderline objects** where the two grippers should be close in force
(the crossover region), so the selection task is non-trivial.

Realistic sprint target: aim for ~130 objects × 2 grippers; if throughput slips,
60–100 objects × 2 with good coverage still supports the study.

## Image capture

Object images are captured with the **Orbbec camera's RGB stream**, which the
OS exposes as a **standard USB (UVC) webcam** — so it is read with
`cv2.VideoCapture`, *not* `pyorbbecsdk`. The SDK is only needed for the depth
stream, which the current pipeline does not use; do **not** install or import
`pyorbbecsdk` for image collection. If a camera call ever fails with
`No module named 'pyorbbecsdk'`, something is wrongly going through the depth
path (`force_prediction.hardware.OrbbecCamera`) instead of `cv2.VideoCapture`.

Two capture tools, both `cv2.VideoCapture` based:

1. **Bulk dataset images** — `scripts/collect_images.py`
   ```bash
   VENV=/Users/premshah/Desktop/Robotics/GSET/env/bin/python
   $VENV scripts/collect_images.py                 # camera 0 -> data/real_chosen_objects/
   $VENV scripts/collect_images.py --camera 1 --width 1920 --height 1080
   ```
   Live OpenCV window: **SPACE** saves `image_NNNN.png`, **q/ESC** quits.
   Use the `--camera` index to pick the Orbbec feed if it is not device 0.

2. **Contact-area test capture** — the **"Contact Area"** tab in
   `streamlit run app.py`. Live preview + Capture, with a **camera index**
   selector (try 0/1 to find the Orbbec RGB feed). Each capture runs the
   geometric contact model and writes `data/test_contact_area/<name>/` (image
   named `<name>.png`, plus outline overlay, CSV, contact figure, and
   `summary.json`). See `scripts/contact_model/README.md`.

Capture tips: object alone on a plain background (background removal keeps the
largest foreground blob — keep the gripper and clutter out of frame), jaws
closing along the image **x**-axis, and include a known-width fiducial in the
scene if you need real millimetre scale (`px per mm`) for the contact model.

### Scale calibration (`px per mm`) — `scripts/calibrate_scale.py`

The contact model is in millimetres, so it needs a pixels-per-millimetre
scale. A single camera's scale is only valid at **one depth plane**, so the
whole rig hinges on one rule:

> **Every object's front face must sit at the same camera distance as the
> reference you calibrate on.** Tape a line on the table for the object fronts;
> calibrate with a known-size reference on that same line; keep the camera
> level (not tilted) and objects centred.

The scale is stored in **`config.yaml` → `geometry.px_per_mm`** (the single
source of truth; the Contact Area tab loads it as the default `px per mm`).

```bash
$VENV scripts/calibrate_scale.py --camera 0
```
SPACE freezes a frame. Press **s** to segment the object and read its
height/width in px and mm using the config scale. To re-measure the scale,
click two points of known real separation (best: the base and top of a
vertical object of known height — vertical measurement has no depth
ambiguity), press **d**, and type the distance in mm. The tool then **writes
the new value straight into `geometry.px_per_mm` in `config.yaml`**
automatically (only that number changes; comments are preserved) and prints
the old → new update.

Re-measure whenever the camera or the object-placement line moves.

## Dataset format (`data/experiences.jsonl`, one row per object-gripper)
```json
{"object_id": "object_001", "image_path": "data/images/object_001.png",
 "mass_g": 420.0, "roughness_class": 2, "projected_contact_fraction": 0.83,
 "gripper": "gecko", "min_force_n": 1.27, "feasible": true,
 "failed_at_limit_n": null, "semantic_description": "smooth rigid plastic bottle",
 "meta": {"trial_forces_n": [1.27, 1.31, 1.28], "n_trials": 3, "pad_id": "g-01",
          "temp_c": 22.5, "humidity_pct": 40, "date": "2026-07-21"}}
```

## Splitting (must not leak)
Both rows of an object share a fold: `GroupKFold(object_id)` via
`evaluation.make_folds`, frozen to `data/splits.json`. Group near-identical
variants together with the `variant_groups` argument.
