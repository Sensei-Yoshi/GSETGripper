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
- `experiments/` is the readable canonical map with one strategy module per active method.
- `pipeline.py` is deliberately thin: `Pipeline(cfg, "e4").fit(train).predict(query)`.
- `contracts.py` owns shared Pydantic data shapes; do not create ad-hoc response dicts.
- Every learned resource is fit inside the current object-grouped training fold.
- Explicit test-only Gemini fakes and a network guard keep unit tests free of paid calls.

## Active experiments

| ID | Method |
|---|---|
| E1 | Image-only zero-shot VLM response for the active grippers |
| E3 | Semantic-cosine experiential retrieval + VLM response |
| E4 | Semantic + mass retrieval + VLM response |
| E5 | E4 + continuous roughness |
| E6 | E5-ranked surfaces + projected-contact condition evidence |

One active gripper uses `PerGripperPrediction`; both use one `JointGripperPrediction`.
Python always makes the final feasible choice; paired VLM recommendation is stored and
scored separately.

## Locked conventions

- Force means stationary-finger load-cell normal force in newtons; never double it.
- Benchmark predictions are continuous and nonnegative with no analytical upper cap; never
  snap predictions to the collection staircase. The physical rig retains its separate 8 N
  handoff/collection safety guard.
- `retrieval.k` in `config.yaml` is the default used by E3–E6 and the UI.
- `surface_id` identifies a physical contact surface; `condition_id` identifies one
  independently measured condition; baseline `object_id` values remain unchanged.
- Gecko/silicone rows are paired by condition-level `object_id`. The source CSV's
  `split` column defines the canonical train/test holdout, and every sibling condition
  of one `surface_id` must remain on the same side.
- E3–E6 rank `retrieval.k` distinct surfaces and retain up to
  `retrieval.conditions_per_surface` condition observations from each.
- Embeddings contain the semantic contact-region description only. Mass, roughness, and
  projected contact remain explicit measured evidence where enabled by the fixed profile.
- E3 ranking and its VLM payload must contain no query/neighbor mass, roughness, contact,
  physical-score components, or physics estimate.
- Live calls receive the query-object image and fixed written descriptions of only
  the active gripper embodiments; gripper images are not sent.
- E1/E3 never require physical measurements. E4 requires mass, E5 adds continuous
  roughness, and E6 adds projected contact evidence without contact-based neighbor ranking.

## Module map

| File | Role |
|---|---|
| `config.py`, `config.yaml`, `prompts.yaml` | Typed config, methods, prompts, embodiments |
| `experiments/` | Strategy catalog, shared helper, and per-ID implementations |
| `pipeline.py` | Public facade and object-to-query adapter |
| `contracts.py` | Experience, query, joint prediction, selection models |
| `prediction.py` | Single/joint force requests, nonnegative normalization, selector |
| `retrieval.py` | Semantic and hybrid retrieval for E3–E6 |
| `physics.py` | Mock-hardware analytical equations and calibration diagnostics |
| `evaluation.py` | Surface-grouped cross-validation splits and metrics |
| `datasets/` | Dataset discovery, aggregate/object contracts, artifact storage, preparation stages |
| `models/` | Lazy Gemini, rembg background-removal, and Marigold integrations |
| `cache.py` | Dataset-scoped API caches |
| `artifacts.py` | Current saved-run serialization and provenance |
| `datasets/paired_csv.py` | Paired-CSV loading, editing adapter, and experience conversion |
| `benchmarking.py` | Schema-v11 prediction batches and versioned evaluation artifacts |
| `suites.py` / `reporting.py` | Schema-v12 five-condition suites and comparison exports |
| `app.py` / `streamlit_app/` | Stable Streamlit entrypoint and modular tab implementation |

Read `docs/streamlit-architecture.md` before adding or reorganizing UI tabs. It defines the
shared context, tab registry, reusable prediction components, widget-key rules, and smoke tests.

## Data and provenance

Every non-hidden direct directory under `data/`, except `cache`, is exposed by the global
Streamlit Dataset selector. Dataset-dependent code must use `AppContext.dataset.paths` or
the dataset runtime config, not hard-coded dataset paths. Each `DatasetObject` always has
primary-image/description/embedding attributes, a train/test split, an optional `image_2`
geometry view, and optional mass, roughness, projected-contact-fraction, and gripper outcomes.

Derived descriptors, experience rows, runs, and results stay separate from the source CSV.
API caches live in `data/cache/<dataset>/{generation,embeddings}` and are ignored runtime
data. Current saved runs use schema v10, benchmark artifacts use schema v11, and suites use
schema v12 with prompt, active-gripper, embodiment, generation-input, and truth-snapshot
provenance. Historical artifacts live only in the repository archive and are not loaded.

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
- Compare the five active conditions on frozen real object-grouped splits with confidence
  intervals and subgroup analysis.
- Treat all existing viewer accuracy as synthetic pipeline validation only.
