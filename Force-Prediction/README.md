# Material-Aware Gripper Selection and Force Prediction

Given an unseen object, estimate the continuous minimum stationary-finger normal force
for TPU–Gecko and TPU–silicone grippers, then select the lowest-force feasible option.
The command range is continuous from `0` to `8 N`; predictions are never rounded to a
collection grid.

The active research suite has five explicit methods:

| ID | Method |
|---|---|
| E1 | One joint vision-only zero-shot VLM response |
| E2 | One joint image + measured-input VLM response |
| E4 | One paired-object retrieval + one joint VLM response |
| E5 | Fold-calibrated reduced-order physics |
| E6 | The same E5 physics + a learned semantic residual |

E3 and E3B were removed. E1, E2, and E4 also return an explicit model gripper
recommendation, but Python's lowest-feasible-force rule remains authoritative.

## Quickstart

```bash
make setup
make test
make smoke
```

Stage checks, all offline unless `--live` is passed:

```bash
python scripts/check_hardware.py
python scripts/check_perception.py
python scripts/check_retrieval.py
python scripts/check_physics.py
python scripts/check_prediction.py
python scripts/check_pipeline.py
```

## Streamlit research lab

The viewer uses the 129-object synthetic two-gripper fixture. Known-object runs are
leave-one-object-out; custom queries use the entire experience pool. E4 retrieves the
`k` configured in `config.yaml` (currently five) and makes one structured force request
for both grippers.

```bash
pip install -e ".[viewer,gemini]"
python scripts/prepare_expforce_viewer.py
streamlit run app.py
```

The Single Run page includes a projected-contact checkbox. Disabling it removes contact
from E2/E4 VLM payloads, sets its E4 retrieval weight to zero, and removes the direct E6
residual feature. E5/E6 physics still requires the measured contact fraction.

Use `python scripts/prepare_expforce_viewer.py --live` to download images, checkpoint
contact-region descriptions, and warm reference embeddings. Live calls are content-hash
cached and resumable.

## Experiment runner

```bash
python scripts/run_experiment.py --exp e4 --dry-run
python scripts/run_experiment.py --all --dry-run
```

For live Gemini execution, set `GEMINI_API_KEY` or `GOOGLE_API_KEY`, install the Gemini
extra, and omit `--dry-run`.

## Data collection

```bash
python -m force_prediction.collect --mock --n 40
python -m force_prediction.collect --port /dev/cu.usbmodemXXXX
```

The coarse/fine staircase controls ground-truth search resolution only. It does not limit
the precision of predictions or hardware commands. See
[`docs/data_collection_sop.md`](docs/data_collection_sop.md).

## Code map

| File | Responsibility |
|---|---|
| `config.yaml` | All tunables, model IDs, experiment methods, and prompts |
| `force_prediction/experiments.py` | Canonical experiment catalog and strategy implementations |
| `force_prediction/pipeline.py` | Thin shared `fit`/`predict` facade |
| `force_prediction/contracts.py` | Records, joint predictions, and selection contracts |
| `force_prediction/prediction.py` | Joint VLM request, physics adapter, force clamp, selector |
| `force_prediction/retrieval.py` | Paired-object embeddings and hybrid retrieval |
| `force_prediction/physics.py` | Reduced-order equations, bounded calibration, solver |
| `force_prediction/learning.py` | E6 residual regressor and semantic PCA |
| `force_prediction/evaluation.py` | Grouped splits and force/selection/recommendation metrics |
| `force_prediction/expforce.py` | Synthetic fixture preparation, persistence, and benchmark |
| `app.py` | Streamlit research lab |

See [`docs/experiments.md`](docs/experiments.md) for the scientific comparison and
[`docs/project-context.md`](docs/project-context.md) for the complete current context.
