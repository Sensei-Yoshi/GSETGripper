# Material-Aware Gripper Selection and Force Prediction

Given an unseen object, estimate the continuous minimum stationary-finger normal force
for the globally selected TPU–Gecko and/or TPU–silicone candidates. Paired runs then
select the lowest-force feasible option.
The command range is continuous from `0` to `8 N`; predictions are never rounded to a
collection grid.

The active research suite has four explicit VLM ablation methods:

| ID | Method |
|---|---|
| E1 | Vision-only zero-shot VLM response |
| E2 | Image + measured-input VLM response |
| E3 | Semantic-only experiential retrieval + VLM response |
| E4 | Semantic + sensor-fusion retrieval + VLM response |

Single-candidate runs use a direct per-gripper response. Paired E1–E4 runs also return an
explicit model recommendation, but Python's lowest-feasible-force rule remains authoritative.

## Quickstart

```bash
make setup
make test
```

Gemini stage checks require `GEMINI_API_KEY` or `GOOGLE_API_KEY`; exact requests are cached:

```bash
python scripts/check_hardware.py
python scripts/check_perception.py path/to/object.png
python scripts/check_retrieval.py --confirm-gemini-cost
python scripts/check_physics.py
python scripts/check_prediction.py path/to/object.png
python scripts/check_pipeline.py path/to/object.png --confirm-gemini-cost
```

## Streamlit research lab

The global selector above the tabs discovers each direct dataset folder under `data/`
(excluding `data/cache`). The selection controls the Data Viewer, preparation stages,
Gemini/embedding caches, contact captures, Marigold roughness tests, experience pool, saved
runs, benchmarks, and suites. Image-only datasets support description, embedding, and
Marigold preparation. Datasets with a second view for every object also support batch
surface/contact estimation. Partial datasets can run any experiment whose object-specific
inputs and reference outcomes are ready; they no longer wait for dataset-wide completion.

Known-object runs are leave-one-object-out and custom queries use the entire eligible
experience pool. E3 retrieves by semantic cosine similarity only. E4 uses the configured
semantic + physical hybrid score. The global Gecko/silicone checkboxes request either one
direct per-gripper response or one paired structured response.

Benchmarks use two explicit stages. **Run predictions** saves immutable, truth-free E1–E4
prediction batches for every query-ready object, so E1 works with images alone. **Evaluate &
generate plots** later joins a saved batch to the currently available force labels, evaluates
the labeled subset without model calls, and versions JSON, CSV, PNG, and SVG results. Detailed
prediction/evaluation histories and suite comparisons live in **Runs Viewer**.

```bash
pip install -e ".[viewer,roughness]"
python scripts/prepare_dataset.py --list
python scripts/prepare_dataset.py --dataset MatForce --stages descriptions --confirm-gemini-cost
streamlit run app.py
```

Edit the fixed written gripper descriptions and all prompt text in `prompts.yaml` or the
**Prompts & Embodiments** tab. No gripper images are sent to the VLM. The **Runs Viewer** creates and
resumes saved E1–E4 suites, compares the two grippers in separate panels, inspects
provenance, and exports PNG, SVG, and CSV results. The **Data Viewer** shows the selected
dataset's images, optional measurements/outcomes, descriptions, and embedding status. For a
dataset, its object editor auto-saves nullable measurement/outcome corrections, recalculates
the favored gripper after both labels are complete, and refreshes completed experience
records. Image-folder values are stored under each object's `measurements.json`; names,
images, and descriptors remain read-only in this tab.

The **Marigold Roughness** tab can multi-select dataset images and uploaded images, then run
IID appearance roughness, normal-based topography, or both across the selection. Its default
background-removal pass creates a mask and transparent cutout, crops the RGB object, and
scores an eroded central grasp band that rejects caps and end faces. Dataset results update
in place under `objects/<object_id>/roughness/streamlit/` and
`objects/<object_id>/topography/streamlit/`; uploaded-image results similarly use stable
name-based folders under `test_data/marigold_tests/`. Raw NumPy maps are not persisted, and
models are loaded only after **Run Marigold** is pressed. IID roughness represents BRDF
appearance, not physical height roughness or pad friction.

Raised bumps and grooves use the separate Marigold normals checkpoint. The topographic
analysis removes a locally smoothed base normal field, scores the remaining angular variation
inside the same grasp band, and stores PNG diagnostics rather than NumPy arrays. It is
available in the Streamlit tab and from the batch CLI:

```bash
python scripts/analyze_topography.py --dataset Matforcedata --objects lechee orange
```

The reported topographic score is an uncalibrated monocular-image proxy. Preserve its raw angular
statistics and calibrate it against measured slip or friction before using it as a physical value.

Global roughness and projected-contact switches sit beside the Dataset selector. Disabling
one removes it from E2/E4 VLM payloads and sets its E4 retrieval weight to zero before the
remaining weights are renormalized. E1 and E3 are unaffected because physical fields are
excluded by construction.

The force pipeline uses the recorded continuous `roughness_index` from the LED measurement
system; larger values mean rougher surfaces. It is separate from the Marigold appearance and
topography artifacts. Retrieval compares two indices with an exponential distance kernel whose
`characteristic_scale` is configured in `config.yaml`. Legacy 1–5 classes remain visible for
provenance but are never converted into or substituted for the numerical index.

Use `scripts/prepare_dataset.py --dataset <folder> --stages <stage...>` to run only
`descriptions`, `embeddings`, or `experiences`. Prerequisites are automatic, but downstream
stages are opt-in. For example, the MatForce command above makes only Gemini descriptions;
they appear in Data Viewer immediately on the next rerun. Gemini calls are content-hash cached
under `data/cache/<dataset>/{generation,embeddings}` and resumable. The legacy flat files in
`data/cache` remain readable as Exp-Force entries. See
[`docs/streamlit-architecture.md`](docs/streamlit-architecture.md) for the full contract.

## Experiment runner

```bash
python scripts/run_experiment.py --exp e4 --confirm-gemini-cost
python scripts/run_experiment.py --all --confirm-gemini-cost
```

E1–E4 use Gemini generation and E3/E4 use Gemini embeddings. Cached identical requests
make no new call.

## Data collection

```bash
python -m modules.collect --mock --dataset mock --n 40 --confirm-gemini-cost
python -m modules.collect --dataset collected --port /dev/cu.usbmodemXXXX --confirm-gemini-cost
```

Mock collection simulates hardware but still uses Gemini for object descriptions. The
coarse/fine staircase controls ground-truth search resolution only. It does not limit
the precision of predictions or hardware commands. See
[`docs/data_collection_sop.md`](docs/data_collection_sop.md).

## Code map

| File | Responsibility |
|---|---|
| `config.yaml` / `prompts.yaml` | Numerical tunables/methods and editable prompts/embodiments |
| `modules/experiments/` | Canonical per-experiment strategies plus shared helpers |
| `modules/pipeline.py` | Thin shared `fit`/`predict` facade |
| `modules/contracts.py` | Records, joint predictions, and selection contracts |
| `modules/prediction.py` | Joint VLM request, force clamp, and selector |
| `modules/retrieval.py` | Paired-object embeddings and hybrid retrieval |
| `modules/physics.py` | Reduced-order equations, bounded calibration, solver |
| `modules/evaluation.py` | Grouped splits and force/selection/recommendation metrics |
| `modules/datasets/` | Dataset catalog, object/artifact models, storage, and preparation stages |
| `modules/cache.py` | Dataset-isolated response caches plus Exp-Force legacy read-through |
| `modules/expforce.py` | Synthetic fixture preparation and saved single-run provenance |
| `modules/benchmarking.py` | Immutable prediction batches and versioned truth evaluation |
| `modules/suites.py` / `reporting.py` | Two-stage E1–E4 suites and paper-ready comparisons |
| `modules/models/` | Gemini, background-removal, and Marigold model adapters |
| `app.py` / `streamlit_app/` | Stable Streamlit entrypoint and modular tab implementation |

See [`docs/experiments.md`](docs/experiments.md) for the scientific comparison and
[`docs/project-context.md`](docs/project-context.md) for the complete current context.
