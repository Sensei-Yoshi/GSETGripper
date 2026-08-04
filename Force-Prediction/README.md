# Material-Aware Gripper Selection and Force Prediction

Given an unseen object, estimate the continuous minimum stationary-finger normal force
for the globally selected TPU–Gecko and/or TPU–silicone candidates. Paired runs then
select the lowest-force feasible option.
Benchmark force estimates are continuous, nonnegative, and have no analytical upper cap;
the optional physical serial path retains a separate 8 N actuation safety guard.

The active research suite has five explicit VLM ablation methods with stable IDs:

| ID | Method |
|---|---|
| E1 | Vision-only zero-shot VLM response |
| E3 | Semantic-only experiential retrieval + VLM response |
| E4 | Semantic + mass retrieval + VLM response |
| E5 | E4 + continuous roughness |
| E6 | E5-ranked surfaces + projected-contact condition evidence |

Single-candidate runs use a direct per-gripper response. Paired runs also return an
explicit model recommendation, but Python's lowest-feasible-force rule remains authoritative.

## Quickstart

```bash
make setup
make test
```

Gemini stage checks require `GEMINI_API_KEY` or `GOOGLE_API_KEY`; exact requests are cached:

```bash
python scripts/check_perception.py path/to/object.png
python scripts/check_retrieval.py --confirm-gemini-cost
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
experience pool. E3 retrieves by semantic cosine similarity only. E4 adds mass to ranking,
E5 adds continuous roughness. E6 keeps E5's surface ranking and adds projected contact
as query and controlled same-surface condition evidence. The global Gecko/silicone checkboxes request either one
direct per-gripper response or one paired structured response.

Single Run can optionally send the deterministic selected result to the physical rig.
Enable **Send selected force to gripper after run**, choose a detected serial port, and
run the pipeline. The app waits for the firmware handshake, sends `SELECT` for the chosen
gripper, then sends its `FORCE <newtons>` command. This command actuates the existing
force-seek/lift firmware; prepare the rig before enabling it.

Benchmarks use two explicit stages and the dataset CSV's `split=train/test` assignment.
**Run predictions** saves immutable, truth-free prediction batches for query-ready test
objects; E3–E6 use only train rows as references. **Evaluate & generate plots** later joins a
saved batch to the currently available force labels, evaluates the labeled subset without model
calls, and versions JSON, CSV, PNG, and SVG results. Detailed prediction/evaluation histories and
suite comparisons live in **Runs Viewer**. Its benchmark **Object Inspector** dropdown opens any
saved batch row in the same rich image/input/prediction layout used for a Single Run, scored only
against the explicitly selected saved evaluation version.

```bash
pip install -e ".[viewer,roughness]"
python scripts/prepare_dataset.py --list
python scripts/prepare_dataset.py --dataset MatForce --stages descriptions --confirm-gemini-cost
streamlit run app.py
```

Edit the fixed written gripper descriptions and all prompt text in `prompts.yaml` or the
**Prompts & Embodiments** tab. No gripper images are sent to the VLM. The **Runs Viewer** creates and
resumes saved five-condition suites, compares the two grippers in separate panels, inspects
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

E4–E6 are fixed nested ablations. E4 exposes mass, E5 adds continuous roughness, and E6 adds
projected contact as VLM evidence without using it for cross-object surface ranking. E4/E5
condition variants are visibility-filtered so hidden measurements cannot explain conflicting
force labels. Disabled
terms receive zero retrieval weight and the remaining weights are renormalized. E1 and E3
exclude physical fields by construction.

The force pipeline uses the recorded continuous `roughness_index` from the LED measurement
system; larger values mean rougher surfaces. It is separate from the Marigold appearance and
topography artifacts. Retrieval compares two indices with an exponential distance kernel whose
`characteristic_scale` is configured in `config.yaml`.

Use `scripts/prepare_dataset.py --dataset <folder> --stages <stage...>` to run only
`descriptions`, `embeddings`, or `experiences`. Prerequisites are automatic, but downstream
stages are opt-in. For example, the MatForce command above makes only Gemini descriptions;
they appear in Data Viewer immediately on the next rerun. Gemini calls are content-hash cached
under `data/cache/<dataset>/{generation,embeddings}` and resumable. See
[`docs/streamlit-architecture.md`](docs/streamlit-architecture.md) for the full contract.

## Experiment runner

```bash
python scripts/run_experiment.py --exp e6 --confirm-gemini-cost
python scripts/run_experiment.py --all --confirm-gemini-cost
```

All five conditions use Gemini generation and E3–E6 use Gemini embeddings. Cached identical requests
make no new call.

## Code map

| File | Responsibility |
|---|---|
| `config.yaml` / `prompts.yaml` | Numerical tunables/methods and editable prompts/embodiments |
| `modules/experiments/` | Canonical per-experiment strategies plus shared helpers |
| `modules/pipeline.py` | Shared `fit`/`predict` facade and single-gripper force helper |
| `modules/contracts.py` | Records, joint predictions, and selection contracts |
| `modules/prediction.py` | Joint VLM request, nonnegative force normalization, and selector |
| `modules/retrieval.py` | Paired-object embeddings and hybrid retrieval |
| `modules/serial_output.py` | Optional gripper selection and force actuation over serial |
| `modules/evaluation.py` | Grouped splits and force/selection/recommendation metrics |
| `modules/datasets/` | Dataset catalog, object/artifact models, storage, and preparation stages |
| `modules/cache.py` | Dataset-isolated response caches |
| `modules/artifacts.py` | Current run serialization and provenance |
| `modules/datasets/paired_csv.py` | Generic paired-CSV loading and experience conversion |
| `modules/benchmarking.py` | Immutable prediction batches and versioned truth evaluation |
| `modules/suites.py` / `reporting.py` | Two-stage experiment suites and paper-ready comparisons |
| `modules/models/` | Gemini, background-removal, and Marigold model adapters |
| `app.py` / `streamlit_app/` | Stable Streamlit entrypoint and modular tab implementation |

See [`docs/experiments.md`](docs/experiments.md) for the scientific comparison and
[`docs/project-context.md`](docs/project-context.md) for the complete current context.
