# CLAUDE.md — Force-Prediction Agent Onboarding

Read this file first, then `docs/project-context.md` for the complete rationale.

## Purpose

Predict the minimum stationary-finger normal force for a user-selected Gecko, silicone,
or paired candidate set. Paired runs select the lowest-force feasible result. Force is an
object–gripper interaction. The active experiments compare zero-shot and
experience-conditioned VLM prediction.

## Environment

- Repository: `/Users/premshah/Desktop/Robotics/GSET/GSETGripper/Force-Prediction`
- Python environment: `/Users/premshah/Desktop/Robotics/GSET/env`
- API credentials: local `.env` or `GEMINI_API_KEY`; never persist them.

## Architecture

- `config.yaml` owns numerical tunables and method assignments; `prompts.yaml` owns
  editable prompts and the two fixed written gripper embodiment descriptions.
- `experiments/` is the readable canonical map with one strategy module per E1–E4 method.
- `pipeline.py` is deliberately thin: `Pipeline(cfg, "e4").fit(train).predict(query)`.
- `contracts.py` owns shared Pydantic data shapes; do not create ad-hoc response dicts.
- Every learned resource is fit inside the current object-grouped training fold.
- Explicit test-only Gemini fakes and a network guard keep unit tests free of paid calls.

## Active experiments

| ID | Method |
|---|---|
| E1 | Image-only zero-shot VLM response for the active grippers |
| E2 | Image + authoritative measurements VLM response |
| E3 | Semantic-cosine experiential retrieval + VLM response |
| E4 | Semantic + sensor-fusion retrieval + VLM response |

One active gripper uses `PerGripperPrediction`; both use one `JointGripperPrediction`.
Python always makes the final feasible choice; paired VLM recommendation is stored and
scored separately.

## Locked conventions

- Force means stationary-finger load-cell normal force in newtons; never double it.
- Hardware and predictions are continuous from `0` through `8 N`; never snap predictions
  to the collection staircase.
- `retrieval.k` in `config.yaml` is the default used by E3/E4 and the UI.
- `surface_id` identifies a physical contact surface; `condition_id` identifies one
  independently measured condition; baseline `object_id` values remain unchanged.
- Gecko/silicone rows are paired by condition-level `object_id`. The source CSV's
  `split` column defines the canonical train/test holdout, and every sibling condition
  of one `surface_id` must remain on the same side.
- E3/E4 rank `retrieval.k` distinct surfaces and retain up to
  `retrieval.conditions_per_surface` condition observations from each.
- Embeddings contain the semantic contact-region description only. Mass, roughness, and
  optional projected contact remain explicit E4 hybrid-score terms.
- E3 ranking and its VLM payload must contain no query/neighbor mass, roughness, contact,
  physical-score components, or physics estimate.
- Live E1–E4 calls receive the query-object image and fixed written descriptions of only
  the active gripper embodiments; gripper images are not sent.
- E1/E3 never require physical measurements. E2/E4 always require mass and use roughness
  and projected contact only when their global input switches are enabled.

## Module map

| File | Role |
|---|---|
| `config.py`, `config.yaml`, `prompts.yaml` | Typed config, methods, prompts, embodiments |
| `experiments/` | Strategy catalog, shared helper, and per-ID implementations |
| `pipeline.py` | Public facade and object-to-query adapter |
| `contracts.py` | Experience, query, joint prediction, selection models |
| `prediction.py` | Single/joint force requests, continuous clamp, selector |
| `retrieval.py` | Embedding providers and E3 semantic/E4 hybrid retrieval |
| `physics.py` | Mock-hardware analytical equations and calibration diagnostics |
| `evaluation.py` | Surface-grouped cross-validation splits and metrics |
| `datasets/` | Dataset discovery, aggregate/object contracts, artifact storage, preparation stages |
| `models/` | Lazy Gemini, rembg background-removal, and Marigold integrations |
| `cache.py` | Dataset-scoped API caches and legacy Exp-Force read-through |
| `expforce.py` | Viewer data preparation and saved single-run provenance |
| `benchmarking.py` | Schema-v9 prediction batches and versioned evaluation artifacts |
| `suites.py` / `reporting.py` | Two-stage E1–E4 suites and comparison exports |
| `app.py` / `streamlit_app/` | Stable Streamlit entrypoint and modular tab implementation |

Read `docs/streamlit-architecture.md` before adding or reorganizing UI tabs. It defines the
shared context, tab registry, reusable prediction components, widget-key rules, and smoke tests.

## Data and provenance

Every non-hidden direct directory under `data/`, except `cache`, is exposed by the global
Streamlit Dataset selector. Dataset-dependent code must use `AppContext.dataset.paths` or
the dataset runtime config, not hard-coded Exp-Force paths. Each `DatasetObject` always has
primary-image/description/embedding attributes, a train/test split, an optional `image_2`
geometry view, and optional mass, roughness, projected-contact-fraction, and gripper outcomes.

`data/expforce/dataset.csv` is a synthetic 129-object validation fixture, not
physical evidence. Derived descriptors, experience rows, runs, and results stay separate
from the source CSV. API caches live in `data/cache/<dataset>/{generation,embeddings}`; flat
legacy cache files are read-through Exp-Force entries and are not deleted. Single-run artifacts
remain schema v8. New benchmark prediction/evaluation and suite artifacts use schema v9 with
prompt, active-gripper, embodiment, generation-input, and truth-snapshot provenance. Old artifacts
are never rewritten; schema-v8 benchmarks and suites are inspectable but read-only.

## Commands

```bash
VENV=/Users/premshah/Desktop/Robotics/GSET/env/bin/python
$VENV -m pytest
$VENV -m ruff check .
$VENV -m mypy modules
$VENV scripts/run_experiment.py --all --confirm-gemini-cost
$VENV scripts/check_pipeline.py path/to/object.png --confirm-gemini-cost
$VENV scripts/prepare_dataset.py --list
$VENV scripts/prepare_dataset.py --dataset MatForce --stages descriptions --confirm-gemini-cost
$VENV -m streamlit run app.py
```

The installed mypy 2.3.0 currently may exit with its own internal error; preserve the
command in verification and distinguish a tool crash from project diagnostics.

## Remaining scientific work

- Collect and calibrate the real two-gripper dataset under the standardized protocol.
- Validate roughness sensing and projected-contact estimation.
- Tune retrieval weights only inside training folds.
- Compare E1–E4 on frozen real object-grouped splits with confidence
  intervals and subgroup analysis.
- Treat all existing viewer accuracy as synthetic pipeline validation only.
