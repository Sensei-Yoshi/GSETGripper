# Experiment Protocol

The active implementation contains E1–E4. Every condition runs through
`Pipeline(cfg, experiment_id)`. Partially populated datasets use per-experiment eligibility;
cross-experiment reporting uses the common eligible object intersection.

## Primary ablation suite

All four VLM conditions receive the query-object image plus fixed written embodiment
descriptions for Gecko and silicone. These descriptions define the hardware being
predicted and are not object-specific sensor measurements. Gripper images are not sent.

| ID | Object-specific evidence | Retrieval score | Research question |
|---|---|---|---|
| **E1** | Object image only | None | How accurately can the VLM predict both grippers zero-shot? |
| **E2** | Image + mass + optional roughness/contact | None | What is the isolated benefit of measured physical inputs? |
| **E3** | Image + semantic descriptor + paired retrieved outcomes | Semantic cosine only | What is the isolated benefit of experience retrieval modeled after Exp-Force? |
| **E4** | Image + measurements + semantic descriptor + paired retrieved outcomes | Semantic and physical fusion | What is the performance of the complete proposed pipeline? |

E1–E4 each produce both gripper force/feasibility predictions and an explicit VLM
recommendation. The commanded gripper is still selected deterministically in Python as
the lowest-force feasible prediction; recommendation accuracy and agreement with the
selector are reported separately.

## E3: strictly semantic experience retrieval

E3 is the semantic-only retrieval ablation. It generates or reuses the query's visible
contact-region description, embeds that text, and ranks training objects using only

`S(q, i) = cosine(e_q, e_i)`.

The embedding contains semantic contact-interface text only. E3 does not use query or
neighbor mass, roughness, or projected contact fraction in ranking or in the VLM payload.
Each retrieved neighbor exposes only its semantic description, semantic similarity, and
paired observed Gecko/silicone force and feasibility outcomes. It receives no hybrid
score components and no physics estimate. Tests enforce this boundary so later refactors
cannot quietly leak sensor terms into E3.

This is the right role for E3 because it creates two clean, parallel comparisons:

- E2 minus E1 estimates the contribution of measured sensor/context variables.
- E3 minus E1 estimates the contribution of semantic experiential retrieval.
- E4 tests whether their combination is stronger than either source alone.

E4 having the lowest error is a hypothesis, not a guaranteed result. Sensor noise,
retrieval mismatch, prompt sensitivity, or a small experience set can make fusion worse.
The paper should report the measured differences and uncertainty rather than treating the
expected ordering as a premise.

## E4: semantic and sensor-fusion retrieval

E4 uses the same paired-object experience representation as E3, but ranks candidates with
the configured hybrid score:

`S(q, i) = w_s S_sem + w_m S_mass + w_r S_rough + w_a S_contact`.

The query measurements and the corresponding neighbor values are also available to the
VLM so it can interpret mass, roughness, and contact differences. The retrieval score
only ranks candidates; it never computes force. Force evidence comes from the paired
observed outcomes attached to those neighbors. E4 receives no physics prediction.

Roughness and projected contact are independent explicit ablations. When disabled, E2 and
E4 omit the field, E4 gives its retrieval term zero weight, and the remaining hybrid
weights are renormalized. E3 is unchanged because it never uses either term.

## Evaluation and reporting

The force convention is stationary-finger load-cell normal force, never doubled. Model
outputs are continuous from `0` to `8 N`; ground-truth search resolution does not quantize
predictions. Report force MAE, RMSE, median absolute error, threshold accuracy,
feasibility metrics, selection accuracy, infeasible-pick rate, and regret.

The saved E1–E4 suite produces separate Gecko and silicone calibration panels, one panel
per experiment, plus CSV metric and prediction exports. The panels show ground truth on
the horizontal axis and prediction on the vertical axis without an identity line.
Per-object prediction tables support the proposed ground-truth-versus-prediction analysis.
Treat fixture-based results as synthetic pipeline validation until real standardized
grasp trials replace them.

Run a Gemini-backed condition with
`python scripts/run_experiment.py --exp e3 --confirm-gemini-cost`, run all active
conditions with `python scripts/run_experiment.py --all --confirm-gemini-cost`, or
create/resume an E1–E4 suite from the Streamlit **Runs Viewer**.
