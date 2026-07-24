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
- API credentials: local `.env` or `GEMINI_API_KEY`/`GOOGLE_API_KEY`; never persist them.
- The repository is on an iCloud-synced Desktop. If a file unexpectedly appears empty or
  reverts, re-materialize it and verify the working tree before editing.

## Architecture

- `config.yaml` is the single source for tunables, method assignments, and prompts.
- `experiments.py` is the readable canonical map and implementation for E1/E2/E4/E5/E6.
- `pipeline.py` is deliberately thin: `Pipeline(cfg, "e4").fit(train).predict(query)`.
- `contracts.py` owns shared Pydantic data shapes; do not create ad-hoc response dicts.
- Every learned resource is fit inside the current object-grouped training fold.
- Offline mocks and dry-run VLM responses keep tests free of hardware and network calls.

## Active experiments

| ID | Method |
|---|---|
| E1 | One joint image-only zero-shot VLM response and recommendation |
| E2 | One joint image + authoritative measurements VLM response |
| E4 | One paired-object retrieval + one joint VLM response |
| E5 | Bounded calibration of seven reduced-order physics coefficients |
| E6 | Identical E5 calibration + one semantic residual regressor per gripper |

E3 and E3B do not exist. The numbering gap is intentional. Python always makes the final
lowest-feasible-force choice; VLM recommendation is stored and scored separately.

## Locked conventions

- Force means stationary-finger load-cell normal force in newtons; never double it.
- Hardware and predictions are continuous from `0` through `8 N`; never snap predictions
  to the collection staircase.
- `retrieval.k` in `config.yaml` is the default actually used by E4 and the UI.
- Stored experiences are grouped by `object_id`; both gripper rows share every split.
- Query objects must be excluded from their own E4 retrieval pool.
- Embeddings contain the semantic contact-region description only. Mass, roughness, and
  optional projected contact remain explicit hybrid-score terms.
- E5 is calibrated physics, not an unlearned formula. E6 is E5 plus a flexible residual.
- E5 and E6 receive no VLM force prediction and retrieve no neighbor list.
- E4 receives no physics value.

## Module map

| File | Role |
|---|---|
| `config.py` / `config.yaml` | Typed config and all experiment definitions/prompts |
| `experiments.py` | Strategy catalog, fold-local fitting, experiment dispatch |
| `pipeline.py` | Public facade and object-to-query adapter |
| `contracts.py` | Experience, query, joint prediction, selection models |
| `prediction.py` | Joint force request, continuous clamp, physics adapter, selector |
| `retrieval.py` | Embedding providers and E4 paired-object retrieval |
| `physics.py` | Analytical capacity equations, bounded calibration, root solve |
| `learning.py` | E6 residual learner and PCA |
| `evaluation.py` | Object-grouped splits and metrics |
| `expforce.py` | Viewer data preparation, v3 artifacts, legacy provenance, benchmark |
| `app.py` | Streamlit research lab |

## Data and provenance

`data/expforce/dataset_2gripper.csv` is a synthetic 129-object validation fixture, not
physical evidence. Derived descriptors, experience rows, cache entries, runs, and results
stay separate from the source CSV. New run artifacts use schema v3 with
`experiment_method` and `experiment_definition_version`. Old artifacts are never rewritten;
the inspector labels old E5 paired-VLM and old E4 physics runs as legacy meanings.

## Commands

```bash
VENV=/Users/premshah/Desktop/Robotics/GSET/env/bin/python
$VENV -m pytest
$VENV -m ruff check .
$VENV -m mypy force_prediction
$VENV scripts/run_experiment.py --all --dry-run
$VENV scripts/check_pipeline.py
$VENV scripts/prepare_expforce_viewer.py
$VENV -m streamlit run app.py
```

The installed mypy 2.3.0 currently may exit with its own internal error; preserve the
command in verification and distinguish a tool crash from project diagnostics.

## Remaining scientific work

- Collect and calibrate the real two-gripper dataset under the standardized protocol.
- Validate roughness sensing and projected-contact estimation.
- Tune retrieval weights, physics coefficients, and residual hyperparameters only inside
  training folds.
- Compare E1, E2, E4, E5, and E6 on frozen real object-grouped splits with confidence
  intervals and subgroup analysis.
- Treat all existing viewer accuracy as synthetic pipeline validation only.
