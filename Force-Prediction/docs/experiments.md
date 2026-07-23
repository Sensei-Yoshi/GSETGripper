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
| **E5** | ✓ | ✓ | ✓ | – | ✓ | – | **Proposed VLM method:** one paired-object retrieval and one joint VLM request. |
| **E6** | ✓ | – | – | ✓ | – | ✓ | Physics + learned residual. Classical small-data baseline. |
| *opt* | ✓ | – | – | formula-in-prompt | ✓ | – | Ablation: VLM given the literal James formula. Label clearly. |

Reading the design: **E5 vs E3** compares the complete paired-object method with
the same-gripper experiential retrieval baseline;
**E5 vs E4** compares the VLM + experiences with physics alone;
**E3 vs E3b** isolates the VLM's contribution over the retrieval it is given;
**E6 vs E4** isolates what a learned residual adds to raw physics.

## Held-fixed across all conditions
Splits, force convention (stationary-finger load cell, never doubled), force
limit and 0.25 N grid, and `k = 5`. E5 retrieves five paired objects once; E3 and
E3b retrieve five rows per gripper branch. Seed,
Gemini model version, temperature 0. Retrieval weights and physics coefficients
are tuned/fit **inside training folds only** — never on test data.

## Paired-object retrieval (E5)
Each object is ranked once from its shared description, mass, roughness, and
projected contact fraction. The top five objects each carry both the gecko and
silicone force/feasibility labels. One structured Gemini request predicts both
candidate forces from this shared evidence, after which Python selects the lower
feasible force. This avoids sending the same object metadata in two prompts and
keeps the within-object gecko-silicone crossover explicit.

E5 does not calculate or send a physics estimate; physics remains isolated in E4
and E6.

## Expected result shape (hypotheses to test)
E1 < E2 < E3 ≲ E5, with E5 lowest force MAE and lowest regret; E4 competitive on
force but weaker on borderline material decisions; E3b establishes the
retrieval-only floor the VLM must beat. Populate the results table from
`run_experiment.py --all --out results.json` once real data is collected.
