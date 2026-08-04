# Project Context: Material-Aware Gripper Selection and Force Prediction

Last updated: 2026-07-26

This is the comprehensive current context for `GSETGripper/Force-Prediction`.
`config.yaml` is the source of truth for numerical tunables and experiment method
assignments. `prompts.yaml` owns editable prompts and fixed gripper embodiment context.
`modules/experiments/` is the source of truth for executable experiment
behavior. Update this document whenever an experiment meaning, data contract, force
convention, or public interface changes.

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
- Does semantic-only paired empirical experience improve the joint VLM result?
- Does fusing semantic experience with measured physical context improve further?

## 2. Scientific claim boundary

The active 129-object Exp-Force viewer fixture is synthetic pipeline-validation data. Its
images are real inputs, but the two-gripper numeric labels and crossover behavior are
constructed fixtures. Current results can establish that conversion, retrieval, prompting,
selection, evaluation, caching, and persistence work. They cannot establish physical
gripper accuracy or real-world safety.

Publishable conclusions require a real two-gripper dataset collected under one controlled
protocol with grouped held-out objects. Until then, the Streamlit app and result reports
must label viewer metrics as synthetic diagnostics.

## 3. Force convention and optional serial actuation

Every force value uses one convention:

- newtons;
- normal force measured behind the stationary gripper finger;
- never doubled and never summed across both fingers;
- nonnegative, continuous dataset labels and benchmark predictions with no analytical upper cap;
- a separate physical serial-actuation safety range above `0` through `8.0 N`;
- no model-output rounding or snapping.

The safety guard does not cap stored experience labels or offline benchmark predictions.
Single Run may explicitly select the predicted gripper and send its continuous force to the
existing firmware; this is an optional physical side effect, not part of model inference.

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
- optional measured mass in grams;
- optional nonnegative numerical roughness index, where higher means rougher;
- optional projected contact fraction from 0 to 1;
- optional prepared semantic contact-region description.

Measured values are authoritative when an experiment exposes them. E1 and E3 intentionally
hide all measurements from the force-prediction VLM. E4 exposes mass, E5 adds continuous
roughness, and E6 adds projected contact. E3 may use
an image-derived semantic descriptor, but no numeric sensor field may enter its ranking or
retrieved payload.

`projected_contact_fraction` is a dimensionless geometry proxy, not physical surface area in
square millimeters. Its intended approximation is:

```text
a = min(1, h_available / h_pad)
```

with the current nominal pad height configured as `65 mm`.

The standalone Streamlit **Contact Fraction** tab now computes a separate
schema-v2 projected two-pad contact fraction from the RGB silhouette:

```text
f_geometric = (left_side_contact_length + right_side_contact_length)
              / (2 * 106.68 mm)

f_contact = max(0.05, f_geometric) for an antipodal grasp; otherwise 0
```

It assumes constant pad width, which cancels from the ratio, and reports no
absolute mm² area. Contact must be contiguous, within 30 degrees of the jaw
direction, and conformable under a default 20 mm minimum bend radius. This
geometric estimator can populate E6's `projected_contact_fraction` evidence through
the dataset preparation workflow.

### Contact-fraction mathematics

The outline is modeled as a closed planar boundary `p(s)` in millimeters, with
outward unit normal `N(s)` and signed curvature `kappa(s)`. The camera plane is
the longitudinal plane of the gripper: `x` is the jaw-closing direction and
`y` points upward.

For pad length `L = 106.68 mm`, only the top-aligned window
`[object_top - L, object_top]` is active. A left-side point is eligible when
`-N_x >= cos(30°)` and a right-side point is eligible when
`N_x >= cos(30°)`. This is the formal reason horizontal top and bottom surfaces
cannot become green contact paths.

The outwardmost eligible point on each side becomes its first-touch anchor.
The anchor pair is accepted only when its normals are sufficiently opposed:

```text
-N_left dot N_right >= cos(40°)
```

Conformability is controlled by minimum bend radius `R_min`, not an arbitrary
curvature slider. Positive convex curvature must satisfy:

```text
kappa <= 1 / R_min
```

At the default `R_min = 20 mm`, the maximum convex curvature is therefore
`0.05 1/mm`. An exterior rolling disk of radius `R_min` also rejects concave
pockets that a finite-radius pad cannot enter.

Starting at each anchor, contact is one contiguous boundary walk. Angle,
curvature, rolling-disk accessibility, pad-window, and material-length tests
must pass at every step. The first failure stops that direction permanently;
there is no gap bridging or re-contact farther along the outline. Accepted
Euclidean segment lengths are integrated, including a partial final segment
when necessary, and each pad is capped at `L`.

```text
ell_left, ell_right in [0, L]
f_geometric = clip((ell_left + ell_right) / (2L), 0, 1)

f_contact = max(0.05, f_geometric)  if the anchor pair is antipodal
f_contact = 0                       otherwise
```

If the anchors fail the antipodal test, the authoritative fraction is zero.
For a valid antipodal pair, `geometry.minimum_contact_fraction` supplies a
default `0.05` floor representing unavoidable TPU seating contact below the
resolution of the macroscopic green-path model. The geometric fraction and a
`contact_floor_applied` flag are retained separately so the assumption is
visible and can later be calibrated from physical data.
The 10/20/30 mm bend-radius sweep is a sensitivity check; increasing minimum
bend radius must not increase predicted contact on the committed regression
fixtures.

For Python entry points, returned fields, schema-v2 JSON/CSV paths, and safe
integration examples, see [`contact-fraction-integration.md`](contact-fraction-integration.md).

Streamlit displays fixed sensor profiles for E4–E6. E5 and E6 always use continuous
roughness so their ranking and VLM evidence remain definition-locked.

Curvature, seams, ridges, porosity, coatings, contamination, and local contact visibility
currently enter through the semantic descriptor rather than separate numeric columns.

## 5. Dataset catalog, object contract, and active fixture

The application discovers every non-hidden direct folder under `data/` except `data/cache`.
That catalog drives one global **Dataset** dropdown above all tabs. The active selection
applies to Data Viewer, Data Preparation, Gemini descriptions, embeddings, experience
records, contact-fraction captures, caches, saved runs, benchmark results, and suites.

`modules.datasets.Dataset` is the aggregate loaded by the application. Its
`objects` mapping contains `DatasetObject` values with stable attributes:

| Attribute | Meaning |
|---|---|
| `image` | source path, availability, hash, and optional remote URL |
| `image_2` | optional calibrated second view used by surface/contact estimation |
| `split` | canonical `train` reference or held-out `test` assignment |
| `description` | optional structured description plus prompt/model provenance |
| `embedding` | optional embedding metadata and cache key; vectors stay in the disk cache |
| `roughness` | optional dataset-scoped Marigold statistics and provenance |
| `mass_g` | optional measured mass |
| `roughness_index` | optional continuous LED-system roughness measurement; higher is rougher |
| `projected_contact_fraction` | optional dimensionless projected two-pad geometry proxy |
| `gripper_outcomes` | optional Gecko/silicone feasibility and force labels |

The aggregate also exposes `images`, `second_images`, `descriptions`, and `embeddings` convenience mappings,
capability flags, a source fingerprint, and dataset-specific artifact paths. Image-only
folders remain useful for Gemini description generation and Data Viewer analysis even though
force pipelines and benchmark controls are disabled. A validated `dataset.csv`
enables paired-data capabilities.

The persistent force-training contract is one `ExperienceRecord` per measurement-condition–
gripper pair. `surface_id` is the shared physical-object identity, `condition_id` identifies
the independently measured mass/roughness/contact/outcome tuple, and condition-level
`object_id` preserves existing APIs. Siblings share images, descriptor, embedding, and
Marigold diagnostics, but never authoritative condition measurements. `ObjectRecord` pairs
the two active-gripper rows for one condition.

The active paired source is:

```text
data/MatForceFinal/dataset.csv
```

It currently contains 44 condition rows across 40 physical surfaces. Preparation never
mutates the source CSV; generated descriptors,
embeddings, experience rows, runs, and benchmarks live in separate paths. Researchers can
make explicit validated corrections in the Data Viewer. Those saves atomically update the
mass, roughness index, projected contact, force, and feasibility CSV fields, recalculate the
favored gripper, and refresh experience rows and provenance. Names, images, descriptors, and
generated artifact metrics remain read-only there.

Preparation is stage-selectable and resumable:

1. image indexing is an automatic prerequisite;
2. Gemini descriptions can run alone and stop;
3. embeddings add missing descriptions first but do not create experiences;
4. experiences add missing descriptions first and include every completed gripper outcome;
   missing physical measurements and partial paired labels do not block the cache.

Per-object checkpoints live at `data/<dataset>/objects/<object_id>/descriptor.json`, and the
stage manifest lives at `data/<dataset>/preparation_manifest.json`. No downstream stage runs merely because an
upstream stage was selected.

The source CSV's `split` column assigns each physical surface to `train` or `test`; all sibling
conditions must remain together. Benchmark generation predicts only query-ready test objects,
and E3–E6 retrieve only eligible train objects. Predictions are persisted without truth; later
evaluation joins only the current truth-ready subset. Custom counterfactual queries use the full
eligible reference pool.

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
remain explicit numeric retrieval terms. The configured Gemini text embedding model uses
document/query formatting for every production retrieval request.
Exact in-memory search is sufficient for the current corpus, so no vector database is used.

## 7. Active experiment suite

The active stable IDs are E1, E3, E4, E5, and E6.

| ID | Method | Force-generation calls | Meaning |
|---|---|---:|---|
| E1 | `vision_vlm` | 1 | Object-image zero-shot prediction |
| E3 | `semantic_retrieval_vlm` | 1 | Semantic-cosine experiential retrieval, without sensor inputs |
| E4 | `hybrid_retrieval_vlm` | 1 | Semantic + mass retrieval |
| E5 | `hybrid_retrieval_vlm` | 1 | E4 + continuous roughness |
| E6 | `hybrid_retrieval_vlm` | 1 | E5-ranked surfaces + projected-contact condition evidence |

“Force-generation call” means a VLM call that returns force predictions. E3–E6
may need descriptor or embedding calls when semantic data is not already cached. Those
preparation calls are distinct from force generation.

Live calls receive the query-object image and fixed written descriptions of the
globally active gripper embodiments. One target uses `PerGripperPrediction`; two targets
use one `JointGripperPrediction`. No gripper images are sent.

### E1: vision-only zero-shot

E1 sends no object-specific mass, roughness, contact, retrieved examples, or analytical
estimate. Its structured response covers only the active candidates; paired runs also
return `recommended_gripper`.

### E3: semantic-only experience-conditioned VLM

E3 generates or reuses one contact-region description and embedding per surface, then ranks
training surfaces only by `cosine(e_q, e_i)`. It retrieves `retrieval.k` distinct surfaces.
Each surface is sent once with semantic similarity and up to three active-gripper outcome
observations; condition names and measurement fields are omitted.

E3 sends no query or neighbor mass, roughness, projected contact, hybrid score components,
or analytical estimate. This hard boundary models semantic experiential retrieval in the
current experience corpus and makes E3 minus E1 an interpretable estimate of semantic experience
value.

### E4–E6: nested semantic and sensor-fusion experience-conditioned VLM

E4–E6 share E3's grouped representation and calculate semantic similarity once per surface.
E4 ranks surfaces using semantic and mass terms; E5 and E6 use semantic, mass, and continuous
LED roughness. E6 excludes projected contact from cross-object ranking and exposes it instead
as query and controlled same-surface condition evidence. Each surface is ranked by its best
eligible condition and contributes its baseline plus at most two variants whose recorded
changes are fully visible in that experiment.

The score only ranks neighbors; it never computes force. Force evidence comes from the
observed active-gripper outcomes. E4–E6 receive no analytical force prior. The adjacent E4→E5→E6
comparisons isolate the incremental value of roughness and projected contact; improved error
is an empirical hypothesis rather than an assumed result.

## 8. Target-aware VLM response and authoritative selection

With one active gripper, all conditions use `PerGripperPrediction` directly. With both active,
they use `JointGripperPrediction`:

```text
gecko: PerGripperPrediction
silicone: PerGripperPrediction
recommended_gripper: gecko | silicone | none
comparison_evidence: decisive cross-gripper evidence
recommendation_summary: auditable comparison and recommendation
```

Each per-gripper prediction includes compatibility, feasibility, continuous force, visible
surface properties, evidence used, an explicit calculation summary, assumptions and
uncertainty, and a detailed auditable rationale. The implementation rebinds each nested
candidate to the correct gripper and normalizes force to a nonnegative value without an
analytical upper cap.

For paired runs, `SelectionResult` stores both the authoritative Python choice and the raw
model recommendation, plus `recommendation_agrees_with_selector`. This supports direct evaluation
of zero-shot/model gripper classification without allowing free-form model choice to bypass
the force/feasibility rule.

The shared prediction prompt requires a detailed evidence report rather than an unsupported
number. For each gripper it records all material supplied/visible evidence, explicit
equations or scaling, assumptions, missing information, and uncertainty, then connects
those items to the force estimate. The joint response separately records the decisive
cross-gripper comparison. It may use object weight as a sanity check, but must not invent
calibrated coefficients or treat weight as the required gripper normal force.

## 9. Object-level active-gripper retrieval

E3 uses only the semantic term:

```text
S_E3(q, i) = cos(e_q, e_i)
```

Its trace and VLM payload omit every sensor value and physical score term. E4–E6 use
nested subsets of the hybrid score:

For query `q` and reference object `i`:

```text
S(q, i) =
    w_sem     * cos(e_q, e_i)
  + w_mass    * exp(-|ln(m_q) - ln(m_i)| / sigma_mass)
  + w_rough   * exp(-|r_q - r_i| / roughness_characteristic_scale)
```

Current configured defaults are:

| Parameter | Value |
|---|---:|
| semantic weight | 0.50 |
| mass weight | 0.25 |
| roughness weight | 0.20 |
| mass sigma | 0.70 |
| contact sigma | 0.25 |
| `k` | 5 objects |

Weights normalize before scoring. Results are sorted by descending score and stable object
ID, assigned ranks from one through `k`, and expose raw terms plus weighted contributions.
E3–E6 construct retrieval traces. E3 traces expose semantic similarity only; E4–E6
expose only their enabled raw hybrid terms and weighted contributions. Normalized weights
are also sent to the VLM as ranking provenance, not as force-model coefficients.

## 10. Code architecture and public interface

The package is organized by concern, with one module per experiment strategy:

| File | Responsibility |
|---|---|
| `config.yaml` | Numerical tunables and explicit experiment methods |
| `prompts.yaml` | Editable prompts and fixed written embodiment descriptions |
| `modules/config.py` | Typed loading and fail-fast validation |
| `modules/experiments/` | Shared strategy helper, eligibility, catalog, and active strategy modules |
| `modules/pipeline.py` | Shared fit/predict facade and single-gripper force helper |
| `modules/contracts.py` | Experience, query, joint prediction, and selection models |
| `modules/prediction.py` | Joint VLM request, nonnegative force normalization, selector |
| `modules/retrieval.py` | Semantic-only and hybrid paired-object retrieval |
| `modules/serial_output.py` | Optional selected-gripper force actuation over serial |
| `modules/evaluation.py` | Grouped splits and metrics |
| `modules/datasets/` | Dataset catalog, object/artifact models, storage, and stage runner |
| `modules/cache.py` | Dataset-scoped API caches |
| `modules/artifacts.py` | Current saved-run serialization and provenance |
| `modules/datasets/paired_csv.py` | Generic paired-CSV adapter and experience conversion |
| `modules/benchmarking.py` | Truth-free prediction batches and versioned evaluations |
| `modules/suites.py` | Definition-locked, resumable two-stage experiment suites |
| `modules/reporting.py` | Comparison tables, panels, and paper exports |
| `app.py` / `streamlit_app/` | Stable Streamlit entrypoint and modular tab implementation |

The UI module flow and tab-extension contract are documented in
[`streamlit-architecture.md`](streamlit-architecture.md).

The public lifecycle is:

```python
pipe = Pipeline(cfg, "e6").fit(train_records)
selection = pipe.predict(query)
detailed = pipe.predict_detailed(query)
```

`Pipeline` validates the experiment ID and delegates. It does not decode boolean toggle
combinations. Resource loading is method-specific:

- E1: no descriptor, embedding, or retrieval fit;
- E3: descriptor as needed, embedding provider, semantic-only object index, target-aware VLM;
- E4–E6: descriptor as needed, embedding provider, profile-scoped hybrid object index,
  target-aware VLM.

`PipelineRunResult` contains stable experiment ID/method/version, selection, semantic
description, E3–E6 paired retrieval evidence, effective-input declaration, and cache telemetry.

## 11. Configuration and prompt routing

The experiment section is explicit:

```yaml
experiments:
  e1: {method: vision_vlm, prompt: e1}
  e3: {method: semantic_retrieval_vlm, prompt: e3}
  e4: {method: hybrid_retrieval_vlm, prompt: e4}
  e5: {method: hybrid_retrieval_vlm, prompt: e5}
  e6: {method: hybrid_retrieval_vlm, prompt: e6}
```

The active force-prediction instructions are:

```text
prompts.prediction_system
prompts.experiments.e1
prompts.experiments.e3
prompts.experiments.e4
prompts.experiments.e5
prompts.experiments.e6
```

These instructions and the `gecko`/`silicone` written embodiment descriptions live in
`prompts.yaml`; `config.yaml` references that file. Configuration validation
requires exactly the five active IDs, a valid prompt for every method, and no unused
experiment prompt. Researcher tunables such as force range, retrieval
`k`, weights, sigmas, and models must not be shadowed by hidden script constants.

Explicit CLI/UI overrides operate on a copied config and are persisted with the run. The
default neighbor count displayed and executed by Streamlit comes directly from
`config.yaml`; historical artifacts may truthfully show an older value such as seven.

## 12. Streamlit research lab

The application provides:

- a global **Dataset** selector, populated from direct folders under `data/`, that controls
  every dataset-dependent tab and path;
- **Single Run:** known leave-one-out or custom query with per-condition eligibility;
- **Benchmark:** immutable prediction generation plus later partial-truth evaluation;
- **Runs Viewer:** versioned benchmark results with a per-object Single Run-style inspector,
  resumable experiment suite comparison, saved single-run inspection, provenance, separate gripper
  panels, and PNG/SVG/CSV exports;
- **Data Viewer:** active-dataset images, train/test membership, optional measurements/outcomes,
  descriptions, embedding status, and an auto-saving editor for split and nullable measurements
  and outcomes;
- **Prompts & Embodiments:** validate and atomically save prompt text and written gripper
  descriptions;
- **Contact Fraction:** capture an RGB image and estimate the combined projected
  side-contact fraction of both 4.2-inch pads;
- **Data Preparation:** independently selectable, resumable descriptions, embeddings,
  Marigold roughness, second-view surface/contact estimates, and experience records with
  automatic prerequisites and capability gating;
- **Cache Status:** response/cache counts and latest telemetry;
- **Help & Experiments:** current experiment definitions loaded from configuration.

Single Run shows both predicted forces, final selector output, raw VLM recommendation,
and paired retrieval for E3–E6. Measurement controls are enabled only for the fields in the
selected fixed experiment profile.

Changing an image or measured value creates a counterfactual query. The viewer does not
score it against the unchanged source label; it displays a delta from the original query.

## 13. Evaluation

Each experiment generates predictions for every query-ready test object. Evaluation can happen
later and scores only test rows with complete required truth. E3–E6 retrieval references come
only from train rows, and cross-condition reporting uses the common evaluated object
intersection.

Metrics include:

- force MAE, RMSE, median absolute error, and configured threshold accuracy;
- feasibility precision and recall per gripper;
- deterministic selection accuracy and infeasible-pick rate;
- mean, median, and worst selection regret;
- raw VLM recommendation accuracy and agreement with the selector.

Regret is the true force of the selected gripper minus the true force of the oracle gripper.
It distinguishes a small force penalty from a materially poor selection.

## 14. Persistence and schema versions

Single-run artifacts use schema version 10. Benchmark artifacts use schema version 11,
suite manifests use schema version 12, and suite reporting uses version 4. A prediction batch records:

- experiment ID;
- stable `experiment_method`;
- `experiment_definition_version`;
- exact method definition, prompt bundle hash, prompts, and embodiment descriptions;
- source and image hashes;
- model and embedding versions;
- retrieval configuration including `k`;
- exact train/test IDs and split hash;
- query snapshots, selection, evidence, and cache telemetry, but no truth.

Evaluation artifacts record the prediction batch ID, current truth snapshot hash, coverage,
metrics, evaluated rows, and JSON/CSV/PNG/SVG exports. Repeating an unchanged truth snapshot
reuses its evaluation; corrected truth creates a new version. Suite manifests lock prompts,
experiment definitions, model IDs, retrieval settings, inputs, and active grippers while
excluding mutable truth. They checkpoint each prediction batch so E1 can finish before E3–E6
become data-ready.

In Runs Viewer, selecting a benchmark batch exposes an **Object Inspector** dropdown containing
exactly that batch's query rows. The inspector reuses the normal prediction renderer and shows
the saved image and inputs beside force, selection, evidence, neighbor, and formula details.
Its truth and error display come only from the selected evaluation version; unevaluated rows
remain fully inspectable but are explicitly unscored. Image hashes warn when a saved path is
missing or now points to changed bytes.

Historical files are immutable under `GSET-program-archive-results` and are never discovered
or loaded by the application.

## 15. Cache behavior

Gemini generation and embedding calls use content-addressed disk caching. New entries are
separated by both dataset and operation:

```text
data/cache/<dataset>/generation/<content-key>.json
data/cache/<dataset>/embeddings/<content-key>.json
```

Cache keys include the relevant model, prompt, JSON schema, payload, query image, embedding
dimension, and text. Identical requests within one dataset reuse responses; changes create
new keys. Writes use temporary files followed by atomic replacement.

Reference descriptor checkpoints are separate from API cache entries, so quota-interrupted
preparation can resume. Credentials are loaded from environment or local `.env` and are not
written to artifacts.

## 16. Verification expectations

Network-isolated verification with explicit Gemini test fakes must cover:

- only the five stable active IDs are accepted;
- one single- or joint-target force-generation call for every active condition;
- E1 payload contains no measurements or retrieval;
- requests for removed IDs fail as unknown;
- E3 ranks by semantic cosine only and exposes no query/neighbor sensor or physical-score
  terms;
- E4–E6 use configured `k`, exclude the query, retrieve once, and call jointly once;
- E4 uses only mass, E5 adds continuous roughness, and E6 retains E5 surface ranking while
  adding projected contact to query and controlled-condition VLM evidence;
- E1/E3 run without physical measurements while measured conditions report missing inputs;
- nonnegative uncapped benchmark forces and infeasibility behavior;
- selector authority and recommendation disagreement metrics;
- schema-v11 truth-free batches, versioned partial evaluations, and schema-v12 resumable suites;
- comparison panels contain no identity line and export PNG, SVG, and CSV;
- cache reuse, grouped splits, and full benchmark execution through fake Gemini interfaces.

The installed mypy 2.3.0 may currently terminate with its own internal error. Verification
should still invoke it and distinguish that tool crash from source diagnostics.

## 18. Common commands

From `GSETGripper/Force-Prediction`:

```bash
../../env/bin/python -m pytest
../../env/bin/python -m ruff check .
../../env/bin/python -m mypy modules
../../env/bin/python scripts/run_experiment.py --exp e6 --confirm-gemini-cost
../../env/bin/python scripts/run_experiment.py --all --confirm-gemini-cost
../../env/bin/python scripts/prepare_dataset.py --list
../../env/bin/python scripts/prepare_dataset.py --dataset MatForce --stages descriptions --confirm-gemini-cost
../../env/bin/python scripts/prepare_dataset.py --dataset MatForceFinal --stages descriptions embeddings experiences --confirm-gemini-cost
../../env/bin/python -m streamlit run app.py
```

## 19. Invariants

1. Never mix or double force conventions.
2. Never round continuous model output before inference, evaluation, or artifact persistence.
3. Keep each physical surface in one CSV split and restrict E3–E6 benchmark retrieval to train rows.
4. Keep source data separate from generated artifacts.
5. Keep prompts and embodiment context in `prompts.yaml`; keep numerical tunables and
   experiment methods in `config.yaml`.
6. Keep E1 image-only.
7. Keep E3 semantic-only: no measured query/neighbor fields or hybrid physical terms.
8. Keep E4–E6 as nested semantic/mass/roughness evidence ablations; E6 may expose projected
   contact conditions but must not use contact for cross-object ranking.
9. Keep Python selection authoritative and score model recommendation separately.
10. Persist stable method/version/prompt/embodiment provenance.
11. Keep unit tests independent of physical serial ports, credentials, and network access through explicit fakes.
12. Treat current viewer results as pipeline validation only.
13. Scope all dataset-dependent artifacts and caches to the active dataset.
14. Update tests and this document whenever experiment behavior or contracts change.
