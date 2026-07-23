# Project Context: Material-Aware Gripper Selection and Force Prediction

Last updated: 2026-07-23

This is the comprehensive living context for the Force-Prediction project. It
describes the research purpose, the current implementation, the synthetic
validation dataset, the Streamlit research lab, the experiment definitions, and
the decisions made during development. Future work should update this document
when a scientific assumption, public interface, data contract, or experiment
meaning changes.

`config.yaml` remains the source of truth for tunable values and prompts. The
Python implementation remains the source of truth for executable behavior. This
document explains how those pieces fit together and why they are designed this
way.

## 1. Project purpose

The project studies a joint robotics decision:

1. Choose between two compliant grippers for an object.
2. Predict the minimum normal force needed by each gripper to lift that object.
3. Select the feasible gripper with the lower predicted force.

The two gripper embodiments are:

- `gecko`: a TPU fin-ray finger with a mushroom-cap, gecko-inspired dry
  adhesive pad.
- `silicone`: a TPU fin-ray finger with a non-adhesive, high-friction silicone
  pad.

The prediction target is an object-gripper interaction, not an intrinsic object
property. The same object can require different forces with the two grippers.
The system therefore estimates both `F*(object, gecko)` and
`F*(object, silicone)` before making the final selection.

The main proposed method is E5: use measured physical properties and a visual
description to retrieve similar objects with paired gripper outcomes, ask one
VLM request to estimate both forces, and then make the final selection
deterministically in Python.

## 2. Research motivation

This work is positioned between three related ideas:

- Exp-Force motivates experience-conditioned force prediction: provide a model
  with examples of similar past grasps rather than relying only on a zero-shot
  image judgment.
- DeliGrasp motivates using semantic and physical knowledge to inform grasp
  force and material interaction.
- The reduced-order gecko and silicone models are informed by the gripper work
  in James et al., RoboSoft 2026. In this codebase, those models are used as
  classical baselines and as the base model for residual learning.

The research gap is not merely force regression. The system must reason about a
material crossover: gecko adhesion can be favorable on clean, smooth,
nonporous, broad contact patches, while silicone friction can be more robust on
rough, porous, fibrous, contaminated, or geometrically interrupted contact
patches. Mass alone cannot represent this interaction.

The project tests several questions:

- Does measured mass, roughness, and projected contact improve a vision-based
  VLM estimate?
- Does retrieval of prior grasp outcomes improve force prediction?
- Is it better to retrieve paired objects once than to retrieve separate lists
  for each gripper?
- Does a VLM add value over a similarity-weighted retrieval average?
- Does the proposed E5 method outperform calibrated physics or a
  physics-residual model?
- Can the full data, caching, retrieval, inference, evaluation, and reporting
  pipeline run reliably before real data are collected?

## 3. Scientific claim boundary

All data currently used by the Exp-Force viewer must be treated as synthetic
test data. The images are useful pipeline inputs, but the numeric labels and
the current benchmark results are not evidence of real gripper performance.

The current fixture can establish that:

- data validation and conversion work;
- images can be downloaded and described;
- text descriptions can be embedded and cached;
- hybrid retrieval responds to semantic and sensor changes;
- paired gecko/silicone labels reach the prediction model correctly;
- live and offline inference paths execute;
- exact run provenance and benchmark metrics can be persisted;
- repeated requests reuse cached Gemini outputs.

It cannot establish that:

- a predicted force will lift a real object;
- the synthetic gecko-silicone crossover is physically correct;
- the current retrieval weights are optimal;
- the VLM, physics model, or residual model generalizes to real objects;
- any synthetic MAE or selection accuracy is a publishable performance result.

The Streamlit application should continue labeling the data and outputs as
synthetic pipeline validation.

## 4. Force convention and experimental target

Every force value in this project uses one convention:

- Unit: newtons.
- Quantity: normal force measured by the load cell behind the stationary
  gripper finger.
- Do not double the value.
- Do not report the sum of both finger forces.
- Current force limit: `8.0 N`.
- Current label and command grid: `0.25 N`.
- Minimum reportable force: `0.25 N`.
- Nominal successful hold: `3.0 s` after a `50 mm` lift.

For a real object and gripper, ground truth should be the minimum force on the
staircase grid that completes the standardized lift-and-hold protocol. An
infeasible record means the object failed at the configured force limit.

The final selector chooses the lower predicted force among feasible grippers.
If neither is feasible, it returns `none`. The synthetic dataset has exactly one
ground-truth winner for every object. Defensive code still handles a predicted
tie because force quantization can make two model outputs equal even when the
ground-truth labels are different.

## 5. Measured and perceived inputs

The intended query inputs are:

- RGB object image.
- Measured mass in grams.
- Roughness class from 1 through 5.
- Projected contact fraction from 0 through 1.
- Optional precomputed semantic contact-region description.

The measured mass, roughness, and contact fraction are authoritative. A VLM may
use the image to describe material, condition, porosity, coating, curvature,
seams, ridges, and contact-patch visibility, but it must not replace measured
values with visual guesses.

### Roughness

The current ordinal roughness scale is:

| Class | Label |
|---:|---|
| 1 | smoothest |
| 2 | smooth-mild |
| 3 | moderate |
| 4 | rough |
| 5 | roughest |

The synthetic roughness values are plausibility-oriented fixtures, not sensor
measurements. Real work must define and calibrate the roughness measurement
procedure.

### Projected contact fraction

`projected_contact_fraction` is a dimensionless proxy for how much of the
nominal finger pad height can contact the object at the intended grasp band. It
is not a measured surface area in square millimeters. The general geometric
interpretation is:

```text
a = min(1, h_available / h_pad)
```

The configured pad height is currently `65 mm`. The synthetic fixture preserves
varying contact fractions. It does not contain a physical contact-area column.

### Curvature and local geometry

The source CSV does not contain a numeric curvature field. Curvature and other
local geometry can appear in the Gemini contact-region descriptor when visible
in the image. They currently influence semantic similarity through the text
embedding, not an independent numeric similarity term.

## 6. Synthetic Exp-Force validation fixture

The active source is:

```text
data/expforce/dataset_2gripper.csv
```

The file contains one row per object and these columns:

```text
Object
Image
Mass_g
roughness_class
projected_contact_fraction
silicone_force_n
silicone_feasible
gecko_force_n
gecko_feasible
favored_gripper
```

The adapter converts each object into two `ExperienceRecord` rows, one for each
gripper. The two rows share the image, description, mass, roughness, contact
fraction, and object ID. They differ in gripper, force, and feasibility label.

Current validated snapshot:

- 129 unique objects.
- 258 gripper-specific experience rows.
- All 129 objects are feasible for both grippers in the current fixture.
- 87 objects favor gecko and 42 favor silicone.
- There are no equal-force ground-truth pairs.
- All force labels lie on the `0.25 N` grid.
- Mass range: `1 g` to `1384 g`.
- Contact-fraction range: `0.283` to `0.987`.
- Silicone force range: `0.25 N` to `7.0 N`.
- Gecko force range: `0.25 N` to `7.25 N`.
- Current source SHA-256:
  `e0218fe46636ecd32b4f15bfb687fa72a2069cf9840c2da212ccc00813dd0b22`.

The hash is recorded in saved run and benchmark artifacts. If the CSV is
intentionally updated, derived experiences and reported hashes must be
regenerated.

### No 100/29 split

An earlier viewer design proposed 100 reference objects and 29 held-out objects.
That design is no longer active. All 129 objects are visible in the catalog and
all 129 can serve as experiences.

For a normal run on a known dataset object, the selected object is excluded from
retrieval. Its remaining 128 objects form the experience pool. This is
leave-one-object-out evaluation and prevents exact self-retrieval.

For an uploaded image or a sensor-modified counterfactual query, there is no
valid held-out ground truth. All 129 objects can be used as references, and the
original object's labels are displayed only as context rather than scored as
truth for the modified query.

## 7. Descriptor generation

The visual representation is a structured description of the surface region
that the fingers are likely to grip. It is not intended to be a generic caption
of the whole object.

The descriptor prompt is in `config.yaml` under:

```text
prompts.descriptor_system
prompts.descriptor
```

The prompt tells Gemini to:

- inspect the likely opposing contact patches at a centered lateral grasp band;
- distinguish the contact material from labels, wrappers, sleeves, coatings,
  and exposed contents;
- describe visible surface condition, including dust, moisture, oil, wear, or
  uncertainty;
- describe local geometry such as curvature, seams, ridges, edges, lobes, and
  interruptions;
- identify whether the likely contact patch is actually visible;
- state uncertainty instead of inventing hidden material details;
- produce a concise retrieval description focused on discriminating
  pad-to-object interface evidence;
- avoid recommending a gripper or predicting force;
- avoid repeating measured mass, roughness, or contact fraction.

The prompt also explains that the output will be converted into a text embedding
and used to retrieve objects with similar contact interfaces. It includes good
examples so the desired level of specificity is explicit.

The structured `Description` checkpoint includes the retrieval description,
contact region, contact material, visible material, visible condition, local
geometry, patch visibility, and uncertainty. The `description` property used
for embedding composes the relevant structured fields.

## 8. Embedding design

The current system uses text embeddings of the semantic contact-region
description. It does not use image embeddings.

The active live model is configured as:

```text
gemini-embedding-2-preview
output dimensionality: 1536
```

Reference descriptions are embedded as search documents and query descriptions
are embedded in the query role used by the provider. The mock provider uses a
deterministic hash vector for offline tests.

Only the semantic description enters the embedding. Mass, roughness, and
projected contact fraction remain explicit similarity components. This matters
for both interpretability and counterfactual testing: changing a sensor value
can change retrieval without recomputing an unchanged semantic embedding.

Each object descriptor and reference embedding is prepared once per image, not
once per gripper. The resulting vector is shared by the object's gecko and
silicone outcomes. A query also requires only one semantic embedding per run.

At the current corpus size, exact in-memory cosine search is simpler and
sufficient. There is no vector database. A database should only be introduced
if corpus size, team sharing, or remote persistence creates a real need.

## 9. Hybrid retrieval algorithm

For query object `q` and reference object `i`, retrieval uses:

```text
S(q, i) =
    w_sem     * cos(e_q, e_i)
  + w_mass    * exp(-|ln(m_q) - ln(m_i)| / sigma_mass)
  + w_rough   * (1 - |r_q - r_i| / 4)
  + w_contact * exp(-|a_q - a_i| / sigma_contact)
```

where:

- `e` is the semantic text embedding;
- `m` is mass in grams;
- `r` is the ordinal roughness class from 1 to 5;
- `a` is projected contact fraction from 0 to 1.

Current defaults:

| Parameter | Value |
|---|---:|
| `w_sem` | 0.40 |
| `w_mass` | 0.25 |
| `w_rough` | 0.20 |
| `w_contact` | 0.15 |
| `sigma_mass` | 0.70 |
| `sigma_contact` | 0.25 |
| `k` | 5 objects |

Weights are normalized automatically before use. An all-zero weight vector is
invalid. The Streamlit controls allow the weights and sigma values to be changed
for exploratory runs.

Every retrieval result exposes:

- raw semantic cosine similarity;
- raw mass, roughness, and contact similarities;
- normalized weights;
- weighted contribution of each component;
- total hybrid score;
- rank;
- object sensor values;
- paired gecko and silicone force and feasibility labels.

This trace is part of the research output, not merely a UI convenience. It makes
the retrieval behavior inspectable and supports sensitivity analysis.

## 10. Current E5 algorithm

E5 is the proposed experience-conditioned VLM method. Its current implementation
is intentionally one object retrieval and one force-prediction request, not two
independent gripper requests.

### Step 1: Build the query

The pipeline receives the image, authoritative mass, roughness, contact
fraction, and either a prepared descriptor or permission to generate one.

### Step 2: Embed the description once

The semantic contact-region description is embedded once as a query. Sensor
changes do not invalidate this embedding if the image and description are
unchanged.

### Step 3: Retrieve paired objects once

The hybrid score ranks objects, not gripper rows. The top five objects are
returned. Each retrieved object carries both its gecko and silicone outcomes.

This is important because the source object has paired labels even though the
general storage contract uses one row per object-gripper trial. E5 groups those
rows by object before ranking. The model therefore sees within-object gripper
differences directly.

The absence of silicone rows from a top-five list cannot be used as evidence to
choose gecko, because E5's top-five list is not separated by gripper. Every
neighbor includes both outcomes. This avoids confounding object similarity with
which gripper branch happened to be searched.

### Step 4: Make one structured Gemini request

The request includes:

- the object image;
- authoritative mass, roughness class, and projected contact fraction;
- the roughness scale labels;
- the five paired retrieved objects;
- both force and feasibility labels for each neighbor;
- retrieval totals and component breakdowns;
- normalized retrieval weights and sigma constants;
- force limits and force-grid constraints;
- a schema requiring one gecko prediction and one silicone prediction.

The request does not contain a physics prior. E5 is designed to isolate measured
properties plus retrieval plus VLM reasoning from the physics baselines.

The VLM does not make the final gripper selection. It returns two structured
`PerGripperPrediction` objects with predicted force, feasibility,
compatibility/material assessment, and a concise evidence summary. The summary
is limited to a few sentences; it is not hidden chain-of-thought.

### Step 5: Quantize and select deterministically

Each predicted force is clamped to the configured limits and rounded upward to
the `0.25 N` force grid. Python then selects the lower-force feasible candidate.
If the quantized predictions tie, the selector uses predicted material
compatibility and then stable gripper order as a final deterministic fallback.

### Why this replaced two E5 requests

The earlier implementation retrieved and prompted separately for gecko and
silicone. The current design is better aligned with the paired data:

- one query embedding instead of duplicated semantic work;
- one ranked object list instead of two overlapping lists;
- one image and query payload instead of two;
- one Gemini force request instead of two;
- direct visibility of the gecko-silicone difference for every neighbor;
- lower token use and latency;
- less opportunity for inconsistent reasoning between independent calls.

There is no E5B condition. A temporary E5B label was rejected and removed. E5
itself is the single-retrieval, single-request paired architecture. E3 and E3b
retain branch-specific retrieval because they are experiment baselines, not the
current proposed architecture.

## 11. Physics model

The reduced-order model estimates tangential holding capacity as a function of
normal force `N`, roughness class `c`, and projected contact fraction `a`.

For silicone:

```text
T_silicone = alpha_silicone(c) * a * N
```

For gecko:

```text
T_gecko = alpha_gecko(c) * a * N
        + beta(c) * a * N / (N + N50)
```

The first term represents frictional support. The second gecko term represents
a saturating adhesive contribution. The coefficient families are constrained
to smooth forms across roughness classes rather than fitting an unrelated value
for every class.

The solver finds the minimum quantized normal force whose holding capacity
supports the object's weight, subject to the configured force limit. Parameters
must be calibrated only from the training fold during real evaluation.

The current coefficients are a useful synthetic prior and offline mock truth.
They are not yet calibrated evidence for the real grippers.

## 12. Experiment definitions

All conditions use the same `Pipeline`. Their behavior is controlled by toggle
sets in `config.yaml`, which avoids separate experiment-specific application
implementations.

| ID | Inputs and estimator | VLM generation? | Main question |
|---|---|:---:|---|
| E1 | Image/visual semantics only | Yes | How well does a vision-only zero-shot VLM perform? |
| E2 | Image plus measured mass, roughness, and contact | Yes | Do authoritative measurements improve the VLM? |
| E3 | Measured inputs plus branch-specific retrieval | Yes, one call per gripper | Does Exp-Force-style same-gripper retrieval help? |
| E3b | Measured inputs plus branch-specific retrieval average | No | Does the VLM outperform a direct similarity-weighted average? |
| E4 | Measured inputs plus calibrated reduced-order physics | No | How well does physics alone perform? |
| E5 | Measured inputs plus one paired-object retrieval | Yes, one joint call | Does the proposed paired experiential VLM method work best? |
| E6 | Physics plus a supervised learned residual | No | Does learning systematic physics error beat physics alone? |

### E4 versus E6

E4 outputs the calibrated physics estimate directly.

E6 first computes that same physics estimate and then learns a residual:

```text
residual = true_force - physics_force
final_force = physics_force + predicted_residual
```

The current E6 learner is a gradient-boosted tree using log mass, roughness,
projected contact fraction, physics force, and PCA-reduced semantic text
features. E6 may therefore need cached text embeddings, but it does not use
retrieval and does not make a Gemini force-prediction request.

### E5 versus physics

E5 does not calculate or send a physics prior. Physics is isolated in E4 and E6
so the experiment comparisons remain interpretable. A physics value displayed
for E5 would be misleading and should remain unavailable.

## 13. Offline and Live Gemini modes

The viewer exposes `Offline` and `Live Gemini` execution.

Offline mode:

- makes no network requests;
- uses deterministic mock embeddings;
- uses a similarity-weighted paired-neighbor estimate for E5 instead of VLM
  force inference;
- is appropriate for CI, UI testing, and pipeline plumbing checks;
- must not be described as VLM reasoning.

Live Gemini mode:

- allows Gemini text embeddings and structured generation;
- uses prepared contact-region descriptions for known objects;
- performs one E5 query embedding and one joint E5 force request when neither is
  already cached;
- can complete very quickly when requests are cache hits;
- exposes concise per-gripper VLM evidence summaries in the output.

A fast live result does not prove Gemini was called at that moment. It may mean
the exact embedding or generation request was returned from disk cache. Cache
Status and the run's cache telemetry distinguish hits, misses, writes, and live
backend attempts.

## 14. Data Preparation workflow

The Data Preparation page is for building reusable reference artifacts. It is
not a force benchmark and does not predict gripper forces.

The live preparation action performs, resumably:

1. Validate the 129-row source CSV.
2. Download each missing image.
3. Generate one structured Gemini contact-region descriptor per object unless a
   matching descriptor checkpoint already exists.
4. Write each descriptor checkpoint atomically as soon as it is available.
5. Convert the 129 object records into 258 gripper-specific experience rows.
6. Generate and cache one text-only reference embedding per object.
7. Update the preparation manifest after each object and each embedding.

It does not:

- run E5 force inference;
- create an image embedding;
- generate separate descriptions or embeddings for the two grippers;
- alter the source CSV;
- send a physics prior to Gemini.

Preparation is safe to resume after quota or process interruption. Completed
descriptor files and content-addressed embeddings are reused. Missing images are
reported instead of silently treated as normal descriptions.

Current local preparation status at the time of this update:

- status: complete;
- mode: live Gemini;
- descriptors: 129 of 129;
- warmed reference embeddings: 129 of 129;
- experience rows: 258;
- missing images: 0.

## 15. Cache behavior

Gemini generation and embedding results are stored in the content-addressed disk
cache configured by `paths.cache`, currently `data/cache`.

Writes use a temporary file followed by an atomic replace. Cache telemetry
tracks hits, misses, writes, read errors, and live backend attempts.

Generation cache keys include:

- cache schema version;
- VLM model ID;
- system prompt;
- instruction prompt;
- output JSON schema;
- encoded image bytes;
- structured payload.

Embedding cache keys include:

- cache schema version;
- embedding model ID;
- output dimension;
- embedding text;
- image bytes if a caller supplies an image.

The current retrieval pipeline supplies text only, so reference and query
embedding keys do not contain image bytes.

Consequences:

- repeating an identical request makes no additional Gemini backend call;
- changing model, prompt, image, schema, payload, text, or embedding dimension
  creates a new cache key;
- changing only mass, roughness, contact, or retrieval weights does not require
  a new semantic embedding when the description is unchanged;
- changing sensor values does change retrieval scores and the E5 generation
  payload, so the final force request is correctly invalidated;
- changing the descriptor prompt invalidates descriptor reuse through its
  descriptor signature;
- deleting `data/cache` is allowed, but embeddings and force responses will need
  to be regenerated;
- descriptor checkpoints live separately under
  `data/expforce/descriptors`, so deleting only `data/cache` does not delete
  completed Gemini descriptions.

If both the descriptor directory and cache are deleted, live preparation starts
those stages from scratch. API credentials are loaded locally from environment
variables or `.env` and are never written to artifacts or displayed in the UI.

## 16. Streamlit research lab

Run the local application with:

```bash
../../env/bin/python -m streamlit run app.py
```

The normal local URL is `http://localhost:8501/`.

### Single Run

This page is the main interactive pipeline surface.

The left side contains:

- known-object selection across all 129 objects;
- image upload for a custom query;
- mass, roughness, and projected contact controls;
- experiment selection;
- Offline or Live Gemini selection;
- retrieval weights and sigma constants;
- the Run command.

The right side shows:

- selected gripper and force;
- both per-gripper predictions and feasibility;
- concise VLM evidence summaries when applicable;
- physics estimates only for experiments that use physics;
- the top five paired E5 neighbors and similarity breakdown;
- ground truth and errors for an unmodified known object;
- baseline deltas instead of invalid scoring for counterfactual/custom runs.

### 129-Object Benchmark

This page runs leave-one-object-out evaluation across the full fixture. For each
query object, the other 128 objects form the training and retrieval pool.

The benchmark reports force errors, selection metrics, regret, plots, subgroup
breakdowns, and a sortable object table. Results are persisted as JSON and CSV.
Because the dataset is synthetic, these metrics validate software behavior and
reporting, not physical accuracy.

Live benchmarking can be slow or quota-limited on a free Gemini tier. The cache
makes reruns resumable.

### Data Viewer

The Data Viewer has two roles.

The description catalog shows all 129 objects with:

- the actual image;
- object and image identifiers;
- mass;
- roughness class;
- projected contact fraction;
- both gripper forces;
- both feasibility labels;
- favored gripper;
- generated descriptor and its structured fields;
- descriptor source/model/signature and update time;
- image hash and embedding status/model/dimension/hash.

There is no curvature or physical surface-area CSV value to display. Curvature
may appear in descriptor geometry, and projected contact fraction is the current
contact proxy.

The Pipeline Run Inspector displays saved single-run inputs and full outputs in
a layout similar to Single Run. It includes the exact image, parameters,
experiment toggles, execution mode, retrieval configuration, model versions,
ground truth when valid, predictions, retrieval trace, physics outputs when
applicable, cache telemetry, and persisted source/image hashes.

### Data Preparation

This page builds or resumes image, descriptor, experience-row, and reference
embedding artifacts. It also displays manifest progress and failures. It should
be run before live evaluation when the image catalog or descriptor prompt has
changed.

### Cache Status

This page explains cache location and displays preparation/cache state. Use it
to understand why a live run was fast and whether the backend was actually
called.

### Help and Experiments

This page explains the lab workflow, every experiment, what E5 with Live Gemini
means, how to interpret output, the distinction between preparation and
evaluation, and the synthetic-data limitation.

## 17. Ground truth, selection, and metrics

For every object, evaluation compares both predicted gripper forces with their
paired labels and evaluates the selected gripper.

Force metrics include:

- mean absolute error (MAE);
- root mean squared error (RMSE);
- median absolute error;
- fraction within `0.25 N`;
- fraction within `0.5 N`;
- exact force-grid rate where applicable.

Selection metrics include:

- gripper-selection accuracy;
- infeasible-pick rate;
- selection regret.

Regret is:

```text
regret = true force of selected gripper - true force of oracle gripper
```

Regret captures the cost of a wrong selection better than accuracy alone. The
current synthetic fixture always has one oracle gripper. The evaluator retains
tie-aware logic as a general defensive behavior for future datasets, but equal
force rows should not be generated for this fixture.

Uploaded images and changed sensor values are counterfactual/custom queries.
Their original object labels cannot be used as ground truth because the input
has changed. The viewer therefore shows retrieval and prediction deltas from the
unmodified baseline instead of reporting a misleading correctness score.

## 18. Persistence and provenance

Current derived artifacts are kept separate from the source CSV:

```text
data/expforce/dataset_2gripper.csv          source synthetic fixture
data/expforce/images/                       downloaded RGB images
data/expforce/descriptors/<object>.json     per-object descriptor checkpoints
data/expforce/validation_experiences.jsonl  258 derived ExperienceRecord rows
data/expforce/preparation_manifest.json     resumable preparation state
data/cache/<sha256>.json                    embeddings and Gemini responses
data/expforce/runs/<run_id>.json            exact saved single runs
data/expforce/run_images/<sha256>.png        content-addressed uploaded/run images
data/expforce/results/*.json                 benchmark metadata and full traces
data/expforce/results/*.csv                  flat benchmark result tables
```

A saved run records:

- schema version and run ID;
- timestamp;
- source CSV SHA-256;
- leave-one-out or custom protocol;
- experiment and exact toggles;
- Offline or Live Gemini mode;
- VLM and embedding model versions;
- embedding dimension;
- retrieval weights, sigmas, and `k`;
- query values and image hash;
- counterfactual status;
- valid ground truth when available;
- descriptor, predictions, selected gripper, retrieval trace, physics trace,
  and cache telemetry;
- optional baseline result for a counterfactual run.

Benchmark JSON keeps the nested retrieval traces and metadata. Benchmark CSV is
a flatter per-object table intended for sorting and plotting.

## 19. Code architecture

The package is deliberately flat, with one module per concern.

| File | Responsibility |
|---|---|
| `config.yaml` | All tunables, model IDs, prompts, and experiment toggles |
| `force_prediction/config.py` | Typed configuration loading and validation |
| `force_prediction/contracts.py` | Pydantic data contracts, paired object views, predictions, selections |
| `force_prediction/hardware.py` | Real device protocols and physics-backed mock hardware |
| `force_prediction/perception.py` | Structured descriptor generation and depth/contact utilities |
| `force_prediction/llm.py` | The only Gemini SDK boundary, retries, content cache, telemetry |
| `force_prediction/retrieval.py` | Text embedding providers, hybrid score, paired and branch retrieval |
| `force_prediction/physics.py` | Reduced-order models, calibration, and minimum-force solver |
| `force_prediction/learning.py` | E6 physics-residual learner |
| `force_prediction/prediction.py` | Joint/per-gripper VLM estimators, baselines, selector |
| `force_prediction/pipeline.py` | Shared toggle-driven orchestration and `predict_detailed` |
| `force_prediction/evaluation.py` | Force, feasibility, selection, and regret metrics |
| `force_prediction/expforce.py` | Synthetic fixture validation, preparation, run persistence, benchmark |
| `force_prediction/collect.py` | Real/mock staircase collection controller |
| `scripts/run_experiment.py` | E1-E6 experiment runner |
| `scripts/prepare_expforce_viewer.py` | CLI equivalent of resumable data preparation |
| `app.py` | Local Streamlit research lab |

`Pipeline.predict_detailed(...)` returns the selection plus descriptor,
per-gripper predictions, retrieval evidence, physics estimates, and cache
telemetry. `Pipeline.predict(...)` remains the compact public method and returns
the same `SelectionResult`. Tests assert the two entry points make identical
decisions.

The Streamlit app and the batch evaluator call this shared `Pipeline`; they do
not reimplement force prediction in UI-specific code.

## 20. Configuration responsibilities

Anything a researcher may tune should remain in `config.yaml`, including:

- force convention and bounds;
- roughness labels;
- retrieval `k`, weights, and similarity constants;
- embedding provider, model, and dimensionality;
- physics coefficients and calibration bounds;
- residual model settings;
- Gemini model and temperature;
- descriptor and prediction prompts;
- E1-E6 toggle sets;
- evaluation folds and thresholds.

Do not silently hardcode alternate values in the Streamlit app, scripts, or
pipeline. UI overrides should produce an explicit copied config that is saved
with the run.

## 21. Testing and demonstrated behavior

The current offline suite has 34 passing tests. One live Gemini smoke test is
skipped unless explicitly enabled.

Coverage includes:

- source CSV validation and paired conversion;
- use of all 129 objects in the experience pool;
- full 129-object leave-one-out benchmark offline;
- strict E5 exclusion of physics;
- no physics value in VLM payloads;
- paired object deltas and payloads;
- one E5 object retrieval and one joint VLM call;
- agreement between `predict` and `predict_detailed`;
- every similarity component, normalized weights, ranking, and top `k`;
- semantic-only embedding text;
- descriptor and embedding checkpoint resume behavior;
- exact cache reuse and invalidation with a counting fake backend;
- deterministic selection, infeasibility, and predicted tie handling;
- physics monotonicity, quantization, and force-limit behavior;
- grouped train/test splits for the non-viewer experiment runner.

The key one-call test verifies that E5 invokes one paired retrieval and one
Gemini generation for both grippers. The cache tests verify that identical
generation or embedding requests do not make another backend call, while model,
prompt, image, schema, payload, text, and embedding changes invalidate as
expected.

The last full verification also passed Ruff, mypy, and `git diff --check`.

Use:

```bash
../../env/bin/python -m pytest
../../env/bin/python -m ruff check .
../../env/bin/python -m mypy force_prediction
```

The live smoke test is intentionally opt-in:

```bash
RUN_LIVE_GEMINI_TESTS=1 ../../env/bin/python -m pytest tests/test_gemini_live.py
```

## 22. Development decisions and superseded designs

The following decisions are important because older notes or artifacts may
refer to superseded behavior:

- The 100-reference/29-test split was removed. The viewer now uses all 129
  objects with leave-one-object-out evaluation for known queries.
- Ground-truth force ties were removed from the synthetic CSV. Every object has
  one strictly favored gripper.
- Varying projected contact fractions were preserved. They were not replaced by
  a constant.
- The dataset source remains separate from generated descriptors, embeddings,
  runs, and results.
- The descriptor was changed from a generic object caption to a contact-region,
  retrieval-specific structured record.
- Descriptions and embeddings are generated once per object, not once per
  gripper.
- The E5 physics prior was removed. Physics belongs to E4 and E6.
- E5 was changed from two branch retrievals and two force calls to one paired
  object retrieval and one joint force call.
- A temporary E5B experiment was removed. It must not be reintroduced as the
  main architecture under another label.
- E3 and E3b intentionally keep branch retrieval as comparison conditions.
- The viewer now includes Help and Experiments, Data Viewer, Data Preparation,
  Cache Status, Single Run, and 129-Object Benchmark pages.
- Saved single runs preserve enough input, model, retrieval, and hash metadata
  to reproduce or audit what was shown in the UI.

## 23. What is still unknown

The software pipeline is substantially validated, but the research result is
not. Important unresolved questions include:

- How should the roughness sensor be calibrated and mapped to the five classes?
- How should projected contact fraction be measured reproducibly from depth and
  the intended grasp pose?
- Is a scalar contact fraction sufficient, or should real contact area,
  curvature, or local compliance become explicit terms?
- What standardized cleaning, seating, preload, lift, slip, and failure protocol
  should be used for the gecko pad?
- How many repeated trials are needed per object-gripper pair?
- Should ground truth use the minimum successful trial, a robust percentile, or
  a probabilistic success threshold?
- How should fragile objects or object-damage constraints modify the selection
  objective?
- How stable are the retrieval rankings under different weights and Gemini
  descriptor/model versions?
- Does one joint VLM request actually improve accuracy relative to the E3
  branch-specific baseline, beyond reducing tokens and latency?
- Does the semantic embedding add value after controlling for mass, roughness,
  and contact?
- Does E6 generalize better than E5 once enough real labeled data exist?

These are empirical questions. They should be answered with grouped,
object-level held-out real data, not by tuning against the synthetic fixture.

## 24. Transition to real data

The real-data pipeline should preserve the same public contracts wherever
possible:

1. Collect both gripper outcomes for every object under the same protocol.
2. Keep paired rows under one `object_id`.
3. Record actual image, mass, roughness, contact proxy, feasibility, and minimum
   stationary-finger force.
4. Generate one contact-region descriptor and text embedding per object.
5. Freeze object-grouped train/test splits before tuning.
6. Fit retrieval weights, physics parameters, and residual models only inside
   training folds.
7. Compare all E1-E6 conditions under identical splits and force conventions.
8. Report force error, selection accuracy, regret, feasibility behavior,
   confidence intervals, and subgroup performance.
9. Preserve model versions, prompts, source hashes, and cache provenance for
   every reported result.
10. Treat synthetic results only as software prevalidation.

The existing data collection controller, device protocols, mock hardware, and
`docs/data_collection_sop.md` are the starting point. Real firmware pins,
control gains, sensor calibration, and protocol decisions still need to be
finalized.

## 25. Common commands

From `GSETGripper/Force-Prediction`:

```bash
# Install and test
../../env/bin/python -m pip install -e ".[viewer,gemini]"
../../env/bin/python -m pytest

# Build or resume the 129-object live descriptor/embedding catalog
../../env/bin/python scripts/prepare_expforce_viewer.py --live

# Start the research lab
../../env/bin/python -m streamlit run app.py

# Run configured experiments through the shared pipeline
../../env/bin/python scripts/run_experiment.py --exp e5
../../env/bin/python scripts/run_experiment.py --all --dry-run

# Exercise synthetic hardware/data collection
../../env/bin/python -m force_prediction.collect --mock --n 40
```

Live calls require `GEMINI_API_KEY` or `GOOGLE_API_KEY`, normally in a local
git-ignored `.env` file.

## 26. Invariants for future development

Future changes should preserve these rules unless the research design is
explicitly revised:

1. Never mix force conventions or double stationary-finger force.
2. Keep both gripper rows for an object in the same train/test group.
3. Exclude a known query object from its own retrieval pool.
4. Keep source data separate from generated artifacts.
5. Generate one semantic descriptor and one reference embedding per object.
6. Keep measured mass, roughness, and contact as explicit similarity terms.
7. Keep E5 as one paired-object retrieval and one joint Gemini force request.
8. Do not send a physics prior in E5.
9. Make the final gripper decision deterministically from structured candidate
   predictions.
10. Put prompts and researcher-tunable constants in `config.yaml`.
11. Persist source hashes, model versions, prompts/configuration, and retrieval
   traces for reported experiments.
12. Keep offline CI free of Gemini, hardware, and network requirements.
13. Treat all current viewer accuracy as synthetic pipeline validation only.
14. Do not score uploaded or sensor-modified counterfactuals against unchanged
   source labels.
15. Add or update tests whenever a cache key, experiment definition, data
   contract, or algorithmic path changes.

## 27. Current concise system statement

For the proposed E5 condition, the system generates a contact-region description
from the object image and embeds that description as a semantic representation.
It combines semantic similarity with explicit mass, roughness, and projected
contact similarities to retrieve the five most similar reference objects. Each
reference object contributes both its gecko and silicone force outcomes. A
single structured Gemini request uses the query image, authoritative measured
properties, and paired retrieval evidence to predict the minimum force and
feasibility for both grippers. The software then selects the lower-force
feasible gripper deterministically. The current 129-object dataset and all
reported results are synthetic and are used to validate this complete software
flow before the same protocol is applied to real robot data.
