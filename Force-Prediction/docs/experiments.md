# Experiment Protocol

The active implementation contains five conditions with stable IDs. Every condition runs through
`Pipeline(cfg, experiment_id)`. Partially populated datasets use per-experiment eligibility;
cross-experiment reporting uses the common eligible object intersection.

## Primary ablation suite

All five VLM conditions receive the query-object image plus fixed written embodiment
descriptions for the globally active Gecko and/or silicone candidates. These descriptions
define the hardware being predicted and are not object-specific measurements.

| ID | Object-specific evidence | Retrieval score | Research question |
|---|---|---|---|
| **E1** | Object image only | None | How accurately can the VLM predict the active grippers zero-shot? |
| **E2** | Image + semantic descriptor + active-gripper outcomes | Semantic cosine only | What is the isolated benefit of experience retrieval? |
| **E3** | E2 + mass | Semantic + mass | What does mass-conditioned retrieval add? |
| **E4** | E3 + continuous roughness | Semantic + mass + roughness | What is the incremental value of measured roughness? |
| **E5** | E4-ranked experiences + projected contact fraction | Semantic + mass + roughness | What is the incremental value of contact as query and controlled-condition evidence? |

One selected gripper produces one `PerGripperPrediction`; both selected grippers produce
one joint response and an explicit recommendation. Python remains authoritative. Selection,
regret, and recommendation metrics apply only to paired runs.

## E2: strictly semantic experience retrieval

E2 is the semantic-only retrieval ablation. It generates or reuses the query's visible
contact-region description, embeds that text, and ranks training objects using only

`S(q, i) = cosine(e_q, e_i)`.

The embedding contains semantic contact-interface text only. E2 does not use query or
neighbor mass, roughness, or projected contact fraction in ranking or in the VLM payload.
Retrieval ranks distinct physical surfaces. Each surface exposes its semantic description
once, followed by up to `retrieval.conditions_per_surface` active-gripper observations with
no condition names or physical measurements. It receives no hybrid
score components. Tests enforce this boundary so later refactors
cannot quietly leak sensor terms into E2.

This is the right role for E2 because it creates clean comparisons:

- E2 minus E1 estimates the contribution of semantic experiential retrieval.
- E3 minus E2 estimates the contribution of mass-conditioned retrieval.
- E4 minus E3 estimates the incremental contribution of continuous roughness.
- E5 minus E4 estimates the incremental contribution of projected contact without changing the retrieved surface set.

E5 having the lowest error is a hypothesis, not a guaranteed result. Sensor noise,
retrieval mismatch, prompt sensitivity, or a small experience set can make fusion worse.
The paper should report the measured differences and uncertainty rather than treating the
expected ordering as a premise.

## E3–E5: nested semantic and sensor-fusion retrieval

E3–E5 use the same grouped-surface representation as E2. Their surface score draws
nested subsets from the configured hybrid terms:

`S(q, i) = w_s S_sem + w_m S_mass + w_r S_rough`.

E3 enables semantic and mass terms. E4 and E5 both add continuous roughness to surface
ranking, so they retrieve identical physical surfaces for the same query and pool. E5 exposes
projected contact only as query and controlled same-surface condition evidence. Disabled
ranking terms receive zero weight and the remaining terms are renormalized. E4 and E5 always
use the continuous roughness index.

The VLM receives the normalized weights and kernel scales so it can understand why a
neighbor was ranked highly. These values are ranking provenance, not physical coefficients
and not a force equation. The retrieval score only ranks candidates; it never computes
force. Force evidence comes from the paired
observed outcomes attached to those conditions. A surface is ranked by its best eligible
condition; the top `retrieval.k` distinct surfaces are retained. Each surface then contributes
its baseline and at most two controlled variants. E3 admits only mass changes, E4 admits only
mass/roughness changes, and E5 admits mass/roughness/contact changes. Conditions that also
change an experiment-hidden measurement are excluded. None receives a physics prediction.

## Evaluation and reporting

The force convention is stationary-finger load-cell normal force, never doubled. Model
outputs are continuous, nonnegative, and uncapped for benchmark evaluation; ground-truth
search resolution does not quantize predictions. The rig retains a separate physical safety
guard. Report force MAE, RMSE, median absolute error, threshold accuracy,
feasibility metrics, selection accuracy, infeasible-pick rate, and regret.

Benchmark generation is deliberately truth-free. It persists each query input, prediction,
retrieved evidence, target set, and model/prompt provenance under
`results/predictions/`. Evaluation later joins current completed outcomes for the saved active
grippers and writes a version under `results/evaluations/<batch_id>/`; unchanged truth reuses
the existing version. This lets image-only E1 batches be generated before physical trials.
The source CSV's `split` column defines the canonical holdout: only `test` rows are predicted
and scored, while E2–E5 may retrieve only from `train` rows. All sibling conditions of one
physical surface must share the same split. Datasets without test rows retain the
leave-one-surface-out fallback.

The saved five-condition suite produces separate Gecko and silicone calibration panels, one panel
per experiment, plus CSV metric and prediction exports. The panels show ground truth on
the horizontal axis and prediction on the vertical axis without an identity line.
Per-object prediction tables support the proposed ground-truth-versus-prediction analysis.
Treat fixture-based results as synthetic pipeline validation until real standardized
grasp trials replace them.

Run a Gemini-backed condition with
`python scripts/run_experiment.py --exp e2 --confirm-gemini-cost`, run all active
conditions with `python scripts/run_experiment.py --all --confirm-gemini-cost`, or
create or resume a suite from the Streamlit **Runs Viewer**.
