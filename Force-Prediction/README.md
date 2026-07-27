# Material-Aware Gripper Selection and Force Prediction

Given an unseen object, estimate the continuous minimum stationary-finger normal force
for TPU–Gecko and TPU–silicone grippers, then select the lowest-force feasible option.
The command range is continuous from `0` to `8 N`; predictions are never rounded to a
collection grid.

The active research suite has six explicit methods. E1–E4 are the primary VLM ablation:

| ID | Method |
|---|---|
| E1 | One joint vision-only zero-shot VLM response |
| E2 | One joint image + measured-input VLM response |
| E3 | Semantic-only experiential retrieval + one joint VLM response |
| E4 | Semantic + sensor-fusion paired retrieval + one joint VLM response |
| E5 | Fold-calibrated reduced-order physics |
| E6 | The same E5 physics + a learned semantic residual |

E1–E4 also return an explicit model gripper
recommendation, but Python's lowest-feasible-force rule remains authoritative.

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
surface/contact estimation; the paired Exp-Force fixture additionally supports force
pipelines and evaluation.

For paired data, known-object runs are leave-one-object-out and custom queries use the
entire experience pool. E3 retrieves by semantic cosine similarity only. E4 uses the
configured semantic + physical hybrid score. Both make one structured force request for
both grippers.

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
paired CSV dataset, its object editor validates and atomically saves measurement/outcome
corrections, recalculates the favored gripper, and refreshes experience
records. Names, images, and descriptors remain read-only in this tab.

The **Marigold Roughness** tab runs the IID appearance model on any active-dataset image or
an uploaded override. Its default background-removal pass creates a mask and transparent
cutout, then limits roughness statistics and displayed intrinsic maps to the foreground. It
saves the input, mask, cutout, albedo, roughness and metallicity images, raw map arrays,
statistics, and provenance under `test_data/marigold_tests/<run-id>/`; history mode can reopen
every saved run. Models are loaded only after **Run Marigold** is pressed.

The Single Run page includes a projected-contact checkbox. Disabling it removes contact
from E2/E4 VLM payloads, sets its E4 retrieval weight to zero, and removes the direct E6
residual feature. E3 is unaffected because sensor fields are excluded by construction;
E5/E6 physics still requires the measured contact fraction.

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
python scripts/run_experiment.py --exp e5
```

E1–E4 use Gemini generation, E3/E4 use Gemini embeddings, and E6 uses Gemini embeddings
with its local residual model. E5 is fully local. Cached identical requests make no new call.

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
| `modules/prediction.py` | Joint VLM request, physics adapter, force clamp, selector |
| `modules/retrieval.py` | Paired-object embeddings and hybrid retrieval |
| `modules/physics.py` | Reduced-order equations, bounded calibration, solver |
| `modules/learning.py` | E6 residual regressor and semantic PCA |
| `modules/evaluation.py` | Grouped split