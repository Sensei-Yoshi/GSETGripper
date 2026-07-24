# Project Context: Material-Aware Gripper Selection and Force Prediction

Last updated: 2026-07-24

This is the comprehensive current context for `GSETGripper/Force-Prediction`.
`config.yaml` is the source of truth for tunables, prompts, and experiment method
assignments. `force_prediction/experiments.py` is the source of truth for executable
experiment behavior. Update this document whenever an experiment meaning, data contract,
force convention, or public interface changes.

## 1. Research purpose

For an unseen object, the system must:

1. estimate the minimum normal force required by the Gecko gripper;
2. estimate the minimum normal force required by the silicone gripper;
3. determine feasibility for each within the hardware limit; and
4. command the lowest-force feasible gripper.

The prediction target is an object–gripper interaction, not an intrinsic property of the
object. A single object can have different minimum forces and feasibility outcomes for the
two pad materials.

The two embodiments are:

- `gecko`: TPU fin-ray finger with a mushroom-cap Gecko-inspired dry-adhesive pad;
- `silicone`: TPU fin-ray finger with a non-adhesive high-friction silicone pad.

Gecko adhesion benefits from intimate contact with clean, dry, smooth, nonporous surfaces.
It may degrade on rough, porous, fibrous, dusty, wet, oily, or interrupted surfaces.
Silicone relies primarily on friction and viscoelastic interaction and may be more robust
when dry adhesion is poor. The useful gripper therefore cannot be inferred from mass or
roughness alone.

The research asks:

- Can a VLM estimate both forces and recommend a gripper zero-shot from an image plus
  embodiment descriptions?
- How much do authoritative mass, roughness, and projected contact help?
- Does paired empirical experience improve the joint VLM result?
- How well does a compact, calibrated analytical model perform?
- Does semantic residual learning improve on exactly that calibrated physics baseline?

## 2. Scientific claim boundary

The active 129-object Exp-Force viewer fixture is synthetic pipeline-validation data. Its
images are real inputs, but the two-gripper numeric labels and crossover behavior are
constructed fixtures. Current results can establish that conversion, retrieval, prompting,
selection, evaluation, caching, and persistence work. They cannot establish physical
gripper accuracy or real-world safety.

Publishable conclusions require a real two-gripper dataset collected under one controlled
protocol with grouped held-out objects. Until then, the Streamlit app and result reports
must label viewer metrics as synthetic diagnostics.

## 3. Force convention and hardware behavior

Every force value uses one convention:

- newtons;
- normal force measured behind the stationary gripper finger;
- never doubled and never summed across both fingers;
- continuous command range from `0` through `8.0 N`;
- no model-output rounding or snapping;
- nominal successful hold of `3.0 s` after a `50 mm` lift.

The hardware accepts arbitrary continuous commands inside this range. The collection
staircase has separate coarse and fine steps because ground-truth search needs a practical
experimental resolution. Those steps do not constrain later predictions or commands.

A feasible record stores the measured minimum successful force. An infeasible record stores
`min_force_n: null` and the force limit at which it failed. A failure at `8 N` must never be
misread as a successful minimum of `8 N`.

The final selector is deterministic Python. It considers model-predicted feasibility and
chooses the candidate with the lowest continuous predicted force. Equal forces are resolved
by compatibility and then stable gripper order. If neither candidate is feasible, it
returns `none`.

## 4. Query inputs and projected contact

The shared query contains:

- object ID and RGB image;
- measured mass in grams;
- ordinal roughness class from 1 to 5;
- projected contact fraction from 0 to 1;
- optional prepared semantic contact-region description.

Measured values are authoritative when an experiment exposes them. E1 intentionally hides
all measurements from the VLM. E2 and E4 expose measurements. E5 and E6 consume measured
values numerically.

`projected_contact_fraction` is a dimensionless geometry proxy, not physical surface area in
square millimeters. Its intended approximation is:

```text
a = min(1, h_available / h_pad)
```

with the current nominal pad height configured as `65 mm`.

Streamlit provides a `Use projected contact fraction` checkbox:

- enabled: contact participates in E2/E4 VLM payloads, E4 retrieval, and the direct E6
  residual features;
- disabled: contact is omitted from E2/E4 payloads, its E4 retrieval weight is zero and
  remaining weights are renormalized, and the standalone E6 residual contact feature is
  removed;
- E5/E6 physics still requires contact because it is part of the analytical capacity
  equation. The UI states this explicitly.

Curvature, seams, ridges, porosity, coatings, contamination, and local contact visibility
currently enter through the semantic descriptor rather than separate numeric columns.

## 5. Data contracts and active fixture

The persistent training contract is one `ExperienceRecord` per object–gripper pair. Both
rows for an object share `object_id`, image, mass, roughness, contact, and semantic
description. `ObjectRecord` groups the two rows for paired retrieval and the selection
oracle.

The active synthetic source is:

```text
data/expforce/dataset_2gripper.csv
```

It contains 129 objects and converts to 258 experience rows. Every current object has one
strict winning gripper. The source CSV remains immutable during preparation; generated
descriptors, embeddings, experience rows, runs, and benchmarks live in separate paths.

Both gripper rows for an object must remain in the same training or test fold. A known query
object is excluded from its own E4 retrieval pool. The viewer uses leave-one-object-out
evaluation for known objects and the full pool for custom counterfactual queries.

## 6. Semantic description and embedding

The descriptor prompt focuses on the likely opposing finger-contact patches. It requests:

- material touching the pads, distinguishing substrate from wrapper, label, coating, or
  exposed contents;
- visible condition such as clean, dusty, wet, oily, or worn;
- local geometry, curvature, seams, ridges, edges, lobes, and interruptions;
- whether the likely patch is actually visible;
- grounded uncertainty rather than invented hidden properties.

The descriptor must not recommend a gripper, predict force, or restate measured sensor
values. One descriptor is generated per object and reused for both gripper labels.

Embeddings contain semantic description text only. Mass, roughness, and projected contact
remain explicit numeric retrieval terms. The live provider uses the configured Gemini text
embedding model with document/query formatting; dry runs use deterministic hash vectors.
Exact in-memory search is sufficient for the current corpus, so no vector database is used.

## 7. Active experiment suite

The active IDs are exactly E1, E2, E4, E5, and E6. E3 and E3B were removed. The numbering
gap is intentional so historical result identifiers are not silently reinterpreted.

| ID | Method | Force-generation calls | Meaning |
|---|---|---:|---|
| E1 | `joint_vlm` | 1 | Image-only zero-shot forces, feasibility, compatibility, and gripper recommendation |
| E2 | `joint_vlm_measured` | 1 | E1 joint response with authoritative measurements |
| E4 | `paired_retrieval_vlm` | 1 | One paired-object retrieval followed by one joint response |
| E5 | `calibrated_physics` | 0 | Fold-calibrated reduced-order equations |
| E6 | `physics_semantic_residual` | 0 | The same E5 estimate plus a learned semantic residual |

“Force-generation call” means a VLM call that returns force predictions. E4 and E6 may
need descriptor or embedding calls when a semantic description/vector is not already
cached. Those preparation calls are distinct from force generation.

### E1: vision-only zero-shot

E1 sends one image and the shared system prompt containing both embodiment descriptions.
It does not send mass, roughness, contact, retrieved examples, or physics. One structured
response contains both gripper predictions and `recommended_gripper`.

E1 therefore tests both zero-shot force estimation and zero-shot embodiment selection.
The VLM recommendation is not the controller decision; Python independently selects the
lowest feasible predicted force and records whether the two decisions agree.

### E2: measured-input zero-shot

E2 uses the same one-call joint schema but additionally sends authoritative mass,
roughness, and contact when contact is enabled. It sends no retrieved examples and no
physics estimate. Comparing E1 with E2 isolates the value of measurements.

E2 does not ask the VLM to estimate friction, adhesion, or other physical constants and
does not run the E5 equation. E1 and E2 are both direct VLM force predictors: E1 must make
rough visual judgments about object scale, mass, surface, and contact, while E2 replaces
the available judgments with measured inputs. The contrast is measurement availability,
not "informal physics" versus "formal physics."

### E4: paired experience-conditioned VLM

E4 generates or reuses one contact-region description, embeds it once, ranks paired
objects once, and retrieves exactly `retrieval.k` neighbors from `config.yaml` unless an
explicit runtime override is recorded. Each neighbor carries both Gecko and silicone
force/feasibility outcomes.

One structured VLM request receives the image, measurements, hybrid retrieval trace,
continuous force constraints, and both outcomes for every neighbor. It returns two
predictions and an explicit recommendation. No physics estimate is calculated or sent.
Python remains the final selector.

E2 and E4 otherwise use the same measured query fields, force constraints, joint response
schema, and deterministic selector. The retrieval score is not a modified physics model:
it only ranks neighbors from semantic cosine similarity and measured-property closeness.
It never computes a force. E4's additional force evidence is the paired observed outcomes
carried by those neighbors.

### E5: calibrated physics

E5 is learned in the system-identification sense. It does not use fixed placeholder
coefficients as its normal evaluation path. Inside each training fold it fits seven bounded,
physically meaningful coefficients by nonlinear least squares, then evaluates fixed
analytical equations.

The holding-capacity models are:

```text
silicone: T(N) = alpha_sil(c) * a * N
gecko:    T(N) = alpha_geo(c) * a * N + beta(c) * a * N/(N + N50)
```

with smooth roughness-dependent coefficient shapes. Silicone fits two parameters. Gecko
fits five. The solver finds the minimum normal force whose holding capacity supports object
weight. Defaults from `config.yaml` are used only when a fold has too few valid samples for
one gripper.

E5 initializes no embedding provider, retrieval index, descriptor, or force VLM. It is best
described as **calibrated physics**, not “unlearned physics” or “pure physics.”

### E6: calibrated physics plus semantic residual

E6 first performs the identical fold-local E5 calibration. For every feasible training row
whose physics solve is feasible, it constructs:

```text
residual_target = measured_min_force - calibrated_physics_force
```

It fits one residual regressor per gripper. Default features are:

- log mass;
- roughness class;
- projected contact when the contact feature is enabled;
- calibrated physics force;
- PCA-compressed semantic embedding.

PCA and each regressor are fit only on the training fold. At inference:

```text
E6_force = clamp(E5_physics_force + predicted_residual, 0, 8)
```

Physics infeasibility gates E6; the residual does not overturn it. If a gripper has no valid
residual training samples, E6 explicitly falls back to its E5 estimate. E6 retrieves no
neighbors and makes no VLM force-generation request.

The E5/E6 comparison answers one controlled question: does flexible semantic residual
learning improve upon the same calibrated equation-constrained baseline?

## 8. Joint VLM response and authoritative selection

E1, E2, and E4 share `JointGripperPrediction`:

```text
gecko: PerGripperPrediction
silicone: PerGripperPrediction
recommended_gripper: gecko | silicone | none
recommendation_summary: concise evidence summary
```

Each per-gripper prediction includes compatibility, feasibility, continuous force, visible
surface properties, and a concise evidence trace. The implementation rebinds each nested
candidate to the correct gripper and continuously clamps force to the configured range.

`SelectionResult` stores both the authoritative Python choice and the raw model
recommendation, plus `recommendation_agrees_with_selector`. This supports direct evaluation
of zero-shot/model gripper classification without allowing free-form model choice to bypass
the force/feasibility rule.

The shared prediction prompt requires a concise evidence and calculation summary rather
than an unsupported number: one complete sentence for Gecko, one for silicone, and one or
two sentences comparing them and explaining the recommendation (three to four sentences
total). When mathematical reasoning is used, the response names the relationship and the
values or scaling applied. It may use object weight as a sanity check, but must not invent
calibrated coefficients or treat weight as the required gripper normal force.

## 9. Hybrid paired-object retrieval

For query `q` and reference object `i`:

```text
S(q, i) =
    w_sem     * cos(e_q, e_i)
  + w_mass    * exp(-|ln(m_q) - ln(m_i)| / sigma_mass)
  + w_rough   * (1 - |r_q - r_i| / 4)
  + w_contact * exp(-|a_q - a_i| / sigma_contact)
```

Current configured defaults are:

| Parameter | Value |
|---|---:|
| semantic weight | 0.40 |
| mass weight | 0.25 |
| roughness weight | 0.20 |
| contact weight | 0.15 |
| mass sigma | 0.70 |
| contact sigma | 0.25 |
| `k` | 5 objects |

Weights normalize before scoring. Results are sorted by descending score and stable object
ID, assigned ranks from one through `k`, and expose raw terms plus weighted contributions.
Only E4 constructs a retrieval trace.

## 10. Code architecture and public interface

The package remains flat and organized by concern:

| File | Responsibility |
|---|---|
| `config.yaml` | Tunables, prompts, and explicit experiment methods |
| `force_prediction/config.py` | Typed loading and fail-fast validation |
| `force_prediction/experiments.py` | Canonical catalog and E1/E2/E4/E5/E6 strategies |
| `force_prediction/pipeline.py` | Thin shared fit/predict facade |
| `force_prediction/contracts.py` | Experience, query, joint prediction, and selection models |
| `force_prediction/prediction.py` | Joint VLM request, physics adapter, clamp, selector |
| `force_prediction/retrieval.py` | Semantic embedding and paired-object retrieval |
| `force_prediction/physics.py` | Equations, calibration, and minimum-force solve |
| `force_prediction/learning.py` | Residual regressors and PCA |
| `force_prediction/evaluation.py` | Grouped splits and metrics |
| `force_prediction/expforce.py` | Fixture preparation, persistence, and benchmark |
| `app.py` | Streamlit research lab |

The public lifecycle is:

```python
pipe = Pipeline(cfg, "e4").fit(train_records)
selection = pipe.predict(query)
detailed = pipe.predict_detailed(query)
```

`Pipeline` validates the experiment ID and delegates. It does not decode boolean toggle
combinations. Resource loading is method-specific:

- E1/E2: no descriptor, embedding, retrieval, or physics fit;
- E4: descriptor as needed, embedding provider, paired-object index, joint VLM;
- E5: calibrated physics only;
- E6: calibrated physics, semantic embedding/PCA, residual models.

`PipelineRunResult` contains stable experiment ID/method/version, selection, semantic
description, E4 paired retrieval evidence, E5/E6 physics trace, and cache telemetry.

## 11. Configuration and prompt routing

The experiment section is explicit:

```yaml
experiments:
  e1: {method: joint_vlm, prompt: e1}
  e2: {method: joint_vlm_measured, prompt: e2}
  e4: {method: paired_retrieval_vlm, prompt: e4}
  e5: {method: calibrated_physics}
  e6: {method: physics_semantic_residual}
```

The active force-prediction instructions are:

```text
prompts.prediction_system
prompts.experiments.e1
prompts.experiments.e2
prompts.experiments.e4
```

Configuration validation requires exactly the five active IDs, a valid prompt for every
VLM method, no prompt for E5/E6, and no unused experiment prompt. Researcher tunables such
as force range, collection steps, retrieval `k`, weights, sigmas, physics bounds, residual
hyperparameters, models, and prompts must not be shadowed by hidden script constants.

Explicit CLI/UI overrides operate on a copied config and are persisted with the run. The
default neighbor count displayed and executed by Streamlit comes directly from
`config.yaml`; historical artifacts may truthfully show an older value such as seven.

## 12. Streamlit research lab

The application provides:

- **Single Run:** known leave-one-out or custom query with all five experiments;
- **129-Object Benchmark:** leave-one-object-out evaluation and saved JSON/CSV;
- **Data Viewer:** descriptor catalog and exact saved-run inspector;
- **Data Preparation:** resumable image download, descriptors, rows, and embeddings;
- **Cache Status:** response/cache counts and latest telemetry;
- **Help & Experiments:** current definitions and prompts loaded from configuration.

Single Run shows both predicted forces, final selector output, raw VLM recommendation when
applicable, recommendation disagreement, physics traces for E5/E6, and paired retrieval
only for E4. The input panel exposes the contact ablation and states that E5/E6 physics
still requires contact.

Changing an image or measured value creates a counterfactual query. The viewer does not
score it against the unchanged source label; it displays a delta from the original query.

## 13. Evaluation

All experiments use identical object-grouped folds. E5 calibration, E6 calibration/PCA/
regressors, and E4 retrieval references are built exclusively from training records.

Metrics include:

- force MAE, RMSE, median absolute error, and configured threshold accuracy;
- feasibility precision and recall per gripper;
- deterministic selection accuracy and infeasible-pick rate;
- mean, median, and worst selection regret;
- raw VLM recommendation accuracy and agreement with the selector for E1/E2/E4.

Regret is the true force of the selected gripper minus the true force of the oracle gripper.
It distinguishes a small force penalty from a materially poor selection.

## 14. Persistence, schema versions, and legacy artifacts

New single-run and benchmark artifacts use schema version 3 and record:

- experiment ID;
- stable `experiment_method`;
- `experiment_definition_version`;
- exact method definition and prediction prompts;
- source and image hashes;
- model and embedding versions;
- retrieval configuration including `k`;
- query, truth when valid, selection, evidence, physics, and cache telemetry.

Historical files are never rewritten or deleted. Before definition version 3, E5 meant the
paired-retrieval VLM method and E4 meant calibrated physics. The inspector infers their old
method from saved toggles and labels them, for example:

```text
Legacy E5 — paired retrieval VLM
Legacy E4 — calibrated physics
```

This prevents a saved old E5 from being misread as the current E5 physics experiment.

## 15. Cache behavior

Gemini generation and embedding calls use content-addressed disk caching. Cache keys include
the relevant model, prompt, JSON schema, payload, image content, embedding dimension, and
text. Identical requests reuse responses; changes create new keys. Writes use temporary
files followed by atomic replacement.

Reference descriptor checkpoints are separate from API cache entries, so quota-interrupted
preparation can resume. Credentials are loaded from environment or local `.env` and are not
written to artifacts.

## 16. Verification expectations

Offline verification must cover:

- only E1/E2/E4/E5/E6 accepted;
- one joint force-generation call for E1/E2/E4;
- E1 payload contains no measurements or retrieval;
- E2 payload contains measurements but no retrieval or physics;
- E4 uses configured `k`, excludes the query, retrieves once, and calls jointly once;
- contact ablation removes contact from relevant payloads/features;
- E5 initializes only calibrated physics;
- E6 uses identical calibrated physics and returns physics plus residual;
- fold-local calibration, PCA, and residual fitting;
- continuous force clamps and infeasibility behavior;
- selector authority and recommendation disagreement metrics;
- v3 artifact metadata and legacy E4/E5 labels;
- cache reuse, grouped splits, and full offline benchmark execution.

The installed mypy 2.3.0 may currently terminate with its own internal error. Verification
should still invoke it and distinguish that tool crash from source diagnostics.

## 17. Transition to real data

Real collection should:

1. test both grippers for every object under the same protocol;
2. keep paired rows under one object ID;
3. record image, mass, calibrated roughness, contact proxy, minimum force, feasibility,
   environment, pad ID, and trials;
4. freeze grouped splits before tuning;
5. fit retrieval weights, physics coefficients, PCA, and residual models inside training
   folds only;
6. compare the five active conditions under identical splits;
7. report uncertainty, subgroup behavior, feasibility errors, force error, selection, and
   regret;
8. preserve exact model, prompt, source, and configuration provenance.

Open scientific questions include contact measurement repeatability, roughness calibration,
pad cleaning/seating protocols, trial aggregation, fragile-object constraints, semantic
embedding value beyond measurements, and whether E6 generalizes better than E5 with enough
real data.

## 18. Common commands

From `GSETGripper/Force-Prediction`:

```bash
../../env/bin/python -m pytest
../../env/bin/python -m ruff check .
../../env/bin/python -m mypy force_prediction
../../env/bin/python scripts/run_experiment.py --exp e4 --dry-run
../../env/bin/python scripts/run_experiment.py --all --dry-run
../../env/bin/python scripts/prepare_expforce_viewer.py --live
../../env/bin/python -m streamlit run app.py
../../env/bin/python -m force_prediction.collect --mock --n 40
```

## 19. Invariants

1. Never mix or double force conventions.
2. Never round continuous model output to collection steps.
3. Keep paired object rows in one split and exclude the query from E4 retrieval.
4. Keep source data separate from generated artifacts.
5. Keep prompts and tunables in `config.yaml`.
6. Keep E1 image-only and E2 retrieval/physics-free.
7. Keep E4 as one paired retrieval and one joint force response with no physics input.
8. Keep E5 as fold-calibrated physics with no VLM, retrieval, or embeddings.
9. Keep E6 based on the identical E5 calibration plus fold-local semantic residuals.
10. Keep Python selection authoritative and score model recommendation separately.
11. Persist stable method/version provenance and preserve legacy artifacts read-only.
12. Keep offline tests independent of hardware, credentials, and network access.
13. Treat current viewer results as synthetic pipeline validation only.
14. Update tests and this document whenever experiment behavior or contracts change.
