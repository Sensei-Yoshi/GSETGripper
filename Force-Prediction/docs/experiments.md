# Experiment Protocol

All experiments run through the **single** `Pipeline` (`force_prediction/pipeline.py`)
under **identical frozen GroupKFold splits** (`data/splits.json`, grouped by
`object_id` so an object's two gripper rows never straddle a fold). Each is a
toggle set in `config.yaml`; run with `python scripts/run_experiment.py --exp eN`.

Every experiment reports, per gripper and overall:
- **Force**: MAE, RMSE, medAE, %within {0.25, 0.5} N, grid-exact rate.
- **Feasibility**: precision/recall of the feasible flag.
- **Selection**: gripper-choice accuracy, infeasible-pick rate, and **regret**
  `R(o) = F*(o, chosen) − F*(o, oracle)` (mean / median / worst).

Regret is the headline selection metric: choosing the "wrong" gripper but losing
0.1 N is far less serious than losing 3 N, and accuracy alone hides that.

## Conditions

| ID | measured | retrieval | paired rows | physics | VLM | residual | Question it answers |
|----|:--------:|:---------:|:-----------:|:-------:|:---:|:--------:|---------------------|
| **E1** | – | – | – | – | ✓ | – | Vision-only zero-shot VLM. Weakest baseline. |
| **E2** | ✓ | – | – | – | ✓ | – | Does giving the VLM *measured* mass/roughness/contact help? |
| **E3** | ✓ | ✓ | – | – | ✓ | – | Exp-Force-style experiential retrieval, extended to 2 grippers. |
| **E3b** | ✓ | ✓ | – | – | – | – | **Pure retrieval** (similarity-weighted avg). Does the VLM beat raw retrieval? |
| **E4** | ✓ | – | – | ✓ | – | – | **Physics-only.** Do we need a VLM once we have calibrated physics? |
| **E5** | ✓ | ✓ | ✓ | ✓ | ✓ | – | **Proposed full method** (adds gecko↔silicone paired deltas). |
| **E6** | ✓ | – | – | ✓ | – | ✓ | Physics + learned residual. Classical small-data baseline. |
| *opt* | ✓ | – | – | formula-in-prompt | ✓ | – | Ablation: VLM given the literal James formula. Label clearly. |

Reading the design: **E5 vs E3** isolates the value of physics + paired rows;
**E5 vs E4** isolates the value of the VLM + experiences over physics alone;
**E3 vs E3b** isolates the VLM's contribution over the retrieval it is given;
**E6 vs E4** isolates what a learned residual adds to raw physics.

## Held-fixed across all conditions
Splits, force convention (stationary-finger load cell, never doubled), force
limit and 0.25 N grid, `k = 5` experiences **per gripper branch**, seed,
Gemini model version, temperature 0. Retrieval weights and physics coefficients
are tuned/fit **inside training folds only** — never on test data.

## Paired-row enhancement (E5)
Each retrieved experience is augmented with the *same object's* force on the
*other* gripper (`other_gripper_min_force_n`). This hands the VLM the local
gecko↔silicone crossover delta for physically similar objects — a signal
Exp-Force cannot use because it has a single embodiment. Ablate by comparing E5
to the same config with `use_paired_rows: false`.

## Expected result shape (hypotheses to test)
E1 < E2 < E3 ≲ E5, with E5 lowest force MAE and lowest regret; E4 competitive on
force but weaker on borderline material decisions; E3b establishes the
retrieval-only floor the VLM must beat. Populate the results table from
`run_experiment.py --all --out results.json` once real data is collected.
