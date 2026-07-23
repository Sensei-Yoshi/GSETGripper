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
