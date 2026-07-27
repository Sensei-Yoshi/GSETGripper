# CLAUDE.md — Force-Prediction Agent Onboarding

Read this file first, then `docs/project-context.md` for the complete rationale.

## Purpose

Predict the minimum stationary-finger normal force for two compliant grippers—TPU–Gecko
dry adhesive and TPU–silicone friction—and select the lowest-force feasible result. Force
is an object–gripper interaction. The project combines experience-conditioned VLM
prediction, calibrated analytical physics, and physics-residual learning.

## Environment

- Repository: `/Users/premshah/Desktop/Robotics/GSET/GSETGripper/Force-Prediction`
- Python environment: `/Users/premshah/Desktop/Robotics/GSET/env`
- API credentials: local `.env` or `GEMINI_API_KEY`; never persist them.

## Architecture

- `config.yaml` owns numerical tunables and method assignments; `prompts.yaml` owns
  editable prompts and the two fixed written gripper embodiment descriptions.
- `experiments/` is the readable canonical map with one strategy module per E1–E6 method.
- `pipeline.py` is deliberately thin: `Pipeline(cfg, "e4").fit(train).predict(query)`.
- `contracts.py` owns shared Pydantic data shapes; do not create ad-hoc response dicts.
- Every learned resource is fit inside the current object-grouped training fold.
- Explicit test-only Gemini fakes and a network guard keep unit tests free of paid calls.

## Active experiments

| ID | Method |
|---|---|
| E1 | One joint image-only zero-shot VLM response and recommendation |
| E2 | One joint image + authoritative measurements VLM response |
| E3 | Semantic-cosine experiential retrieval + one joint VLM response |
| E4 | Semantic + sensor-fusion retrieval + one joint VLM response |
| E5 | Bounded calibration of seven reduced-order physics coefficients |
| E6 | Identical E5 calibration + one semantic residual regressor per gripper |

Python always makes the final
lowest-feasible-force choice; VLM recommendation is stored and scored separately.

## Locked conventions

- Force means stationary-finger load-cell normal force in newtons; never double it.
- Hardware and predictions are continuous from `0` through `8 N`; never snap predictions
  to the collection staircase.
- `retrieval.k` in `config.yaml` is the default used by E3/E4 and the UI.
- Stored experiences are grouped by `object_id`; both gripper rows share every split.
- Query objects must be excluded from their own E3/E4 retrieval pool.
- Embeddings contain the semantic contact-region description only. Mass, roughness, and
  optional projected contact remain explicit E4 hybrid-score terms.
- E3 ranking and its VLM payload must contain no query/neighbor mass, roughness, contact,
  physical-score components, or physics estimate.
- Live E1–E4 calls receive the query-object image and fixed written descriptions of both
  gripper embodiments; gripper images are not sent.
- E5 is calibrated physics, not an unlearned formula. E6 is E5 plus a flexible residual.
- E5 and E6 receive no VLM force prediction and retrieve no neighbor list.
- E4 receives no physics value.

## Module map

| File | Role |
|---|---|
| `config.py`, `config.yaml`, `prompts.yaml` | Typed config, methods, prompts, embodiments |
| `experiments/` | Strategy catalog, shared helper, and per-ID implementations |
| `pipeline.py` | Public facade and object-to-query adapter |
| `contracts.py` | Experience, query, joint prediction, selection models |
| `prediction.py` | Joint force request, continuous clamp, physics adapter, selector |
| `retrieval.py` | Embedding providers and E3 semantic/E4 hybrid retrieval |
| `physics.py` | Analytical capacity equations, bounded calibration, root solve |
| `learning.py` | E6 residual learner and PCA |
| `evaluation.py` | Object-grouped splits and metrics |
| `datasets/` | Dataset discovery, aggregate/object contracts, artifact storage, preparation stages |
| `models/` | Lazy Gemini, rembg background-removal, and Marigold integrations |
| `cache.py` | Dataset-scoped API caches and legacy Exp-Force read-through |
| `expforce.py` | Viewer data preparation, schema-v6 artifacts, provenance, benchmark |
| `suites.py` / `reporting.py` | Resumable E1–E4 suites and comparison exports |
| `app.py` / `streamlit_app/` | Stable Streamlit entrypoint and modular tab implementation |

Read `docs/streamlit-architecture.md` before adding or reorganizing UI tabs. It defines the
shared context, tab registry, reusable prediction components, widget-key rules, and smoke tests.

## Data and provenance

Every non-hidden direct directory under `data/`, except `cache`, is exposed by the global
Streamlit Dataset selector. Dataset-dependent code must use `AppContext.dataset.paths` or
the dataset runtime config, not hard-coded Exp-Force paths. Each `DatasetObject` always has
primary-image/description/embedding attributes, an optional `image_2` geometry view, and optional mass, roughness,
projected-contact-fraction, and gripper outcomes.

`data/expforce/dataset.csv` is a synthetic 129-object validation fixture, not
physical evidence. Derived descriptors, experience rows, runs, and results stay separate
from the source CSV. API caches live in `data/cache/<dataset>/{generation,embeddings}`; flat
legacy cache files are read-through Exp-Force entries and are not deleted. New run artifacts use schema v5 with prompt and
embodiment provenance. Old artifacts are never rewritten;
the inspector labels old E5 paired-VLM and old E4 physics runs as legacy meanings.

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
- Tune retrieval weights, physics coefficients, and residual hyperparameters only inside
  training folds.
- Compare E1–E6 on frozen real object-grouped splits with confidence
  intervals and subgroup analysis.
- Treat all existing viewer accuracy as synthetic pipeline validation only.
