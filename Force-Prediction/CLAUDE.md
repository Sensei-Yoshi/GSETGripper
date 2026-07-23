# CLAUDE.md — Force-Prediction (agent onboarding)

Read this first. It is the current, accurate state of the project. `docs/project-context.md`
is the original *design rationale* (still useful for the "why"), but this file is the
source of truth for the code, decisions, environment, and how to run things.

## What this is
Material-aware **gripper selection + minimum-force prediction** for two soft grippers:
**TPU–gecko** (dry adhesive) vs **TPU–silicone** (friction). Given an unseen object, predict
`F*(o,g)` — the minimum **stationary-finger normal force** to lift it — for each gripper, and
pick the lower-force feasible one. The method sits between **Exp-Force** (experience-conditioned
VLM) and **DeliGrasp** (LLM-inferred physics params → controller). Physics is grounded in
**James et al., RoboSoft 2026** (DOI 10.1109/RoboSoft67810.2026.11522915).

## ⚠️ Environment gotchas — READ BEFORE ANYTHING
- **Repo path:** `/Users/premshah/Desktop/Robotics/GSET/GSETGripper/Force-Prediction`. It has
  been moved several times; if a path is wrong, run
  `find /Users/premshah/Desktop/Robotics -type d -name force_prediction`.
- **iCloud trap (important):** the project sits on the iCloud-synced Desktop. This has caused
  files to (a) get **evicted to 0-byte placeholders** mid-run (→ spurious `ImportError`, empty
  `config.py`) and (b) **revert to older versions**, silently undoing edits. If imports fail on
  "empty" files, or your edits vanish, this is why. Mitigate by re-materializing
  (`find . -name '*.py' -exec cat {} + >/dev/null`) or — far better — **move the repo + venv off
  iCloud** (e.g. `~/Developer/`) or disable iCloud "Optimize Mac Storage" and mark the folder
  "Keep Downloaded".
- **venv:** `/Users/premshah/Desktop/Robotics/GSET/env` (Python 3.11). Install:
  `"$VENV"/bin/python -m pip install -r requirements.txt && "$VENV"/bin/python -m pip install -e .`
- **API key:** put `GEMINI_API_KEY=...` in `.env` at the repo root (git-ignored, auto-loaded by
  `force_prediction.llm.load_dotenv`). Free tier — see quota note below.

## Architecture (mental model)
- **Flat package** `force_prediction/`, one module per concern, no sub-packages.
- **`config.yaml` is the SINGLE source of tuning:** retrieval weights, physics coefficients,
  model IDs, the **prompts**, and the **E1–E6 experiment toggle-sets**. Never hardcode a tunable.
- **One `pipeline.py`** drives every experiment; E1–E6 differ *only* by toggles
  (`use_measured/use_retrieval/use_paired_rows/use_physics/use_vlm/use_residual`). No per-experiment code.
- **Mock-first:** physics-backed mocks in `hardware.py` + `dry_run` LLM stubs → the whole stack
  and all experiments run **offline with no hardware and no API key**.
- **Data flow:** measured mass + LED roughness class + depth-derived contact fraction →
  calibrated physics prior (`physics.py`) + gripper-branched hybrid RAG (`retrieval.py`, with
  paired gecko↔silicone force deltas) → Gemini VLM resolves surface/material semantics
  (`prediction.py`) → deterministic feasible arg-min selector.

## Module map
| File | Role |
|------|------|
| `config.py` / `config.yaml` | typed config loader / all tunables + prompts + experiment toggles |
| `contracts.py` | Pydantic models: `ExperienceRecord`, paired `ObjectRecord`, `Query`, `PerGripperPrediction`, `SelectionResult` |
| `hardware.py` | device `Protocol`s + real serial drivers + physics-backed mocks + `fabricate_records` |
| `physics.py` | James reduced-order model (silicone/gecko), Brent solver, per-fold calibration |
| `retrieval.py` | embedding providers + similarity + hybrid rerank + gripper-branch filter + paired rows |
| `perception.py` | Gemini descriptor + depth height-ratio contact-fraction proxy |
| `llm.py` | single Gemini client (structured JSON + embeddings), disk cache, retries, `.env` loader |
| `prediction.py` | VLM estimator + non-VLM baselines + deterministic `select()` |
| `learning.py` | physics-residual model (E6) |
| `evaluation.py` | GroupKFold splits + force/selection/regret metrics |
| `pipeline.py` | the one toggle-driven orchestration |
| `collect.py` | coarse-to-fine staircase ground-truth collector (real or `--mock`) |
| `scripts/check_*.py` | stand-alone stage debuggers |
| `scripts/run_experiment.py` | run E1–E6 |
| `scripts/expforce_testset.py` | live Gemini POC on the public Exp-Force dataset |
| `app.py` / `expforce.py` | local 100/29 synthetic validation viewer + immutable dataset adapter |

## Decisions locked (do not silently change)
- **VLM:** `config.models.vlm`. Currently `gemini-flash-lite-latest` (best free-tier quota).
  `gemini-flash-latest` / `gemini-3.1-pro-preview` are higher quality if quota allows;
  Exp-Force's best was `gemini-3.1-pro`. `gemini-2.5-flash` is **blocked for new keys** — don't use it.
- **Embedding:** `gemini-embedding-2` (or `-preview`), **semantic-description-only**, **asymmetric retrieval
  format** — stored experiences as documents (`title: none | text: {…}`), queries as
  `task: search result | query: {…}`. Implemented in `GeminiEmbeddingProvider`; `dim: 1536`
  (Matryoshka). Mass, roughness, and contact are explicit hybrid-score terms, not duplicated
  inside the vector. `gemini-embedding-2` has NO `task_type` param (that was `embedding-001`).
- **Physics (James-grounded):** in `config.yaml physics:` — silicone friction **rises** with
  roughness (`alpha_sil_decay` negative); gecko vdW adhesion (`beta`) **collapses** to ~0 by the
  roughest class. Crossover ≈ class 3. Model form matches James Eq. 1/2 in the saturated regime.
  These are a prior/mock-truth; recalibrate per fold on real data.
- **Force convention:** stationary-finger load-cell normal force, newtons, **never doubled**.
  Single `force.limit_n = 8` in config.
- **Retrieval store:** in-memory **exact cosine** over content-hash-cached embeddings — **no
  vector DB** (corpus < 1k). Consider pgvector/Supabase only if the corpus grows large or a
  shared team DB is wanted (that's a sharing win, not a speed win at this scale).
- **Splits:** `GroupKFold` on `object_id` (both gripper rows share a fold), frozen to `splits.json`.

## Where data & embeddings live
```
data/
├── experiences.jsonl   # dataset: 1 ExperienceRecord per line
├── images/             # object RGB
├── splits.json         # frozen GroupKFold
├── cache/<sha256>.json # ← every embedding vector + every VLM response, content-hash keyed
└── expforce/           # synthetic source + 129-object experience pool + viewer artifacts
```
Embeddings are computed on demand, cached to `data/cache/`, and loaded into an in-memory dict
for exact search during a run. No database.

## How to run
```bash
VENV=/Users/premshah/Desktop/Robotics/GSET/env/bin/python
$VENV -m pytest                                             # unit tests (offline)
$VENV -m force_prediction.collect --mock --n 40            # synthetic dataset (no hardware/API)
$VENV scripts/run_experiment.py --all --dry-run           # E1–E6 offline
$VENV scripts/check_pipeline.py                            # full E5 on one object, offline
$VENV scripts/expforce_testset.py --live --limit 6 --k 3  # live Gemini POC (needs .env key)
$VENV tests/test_gemini_live.py                            # single live call smoke
$VENV scripts/prepare_expforce_viewer.py                   # checkpointed descriptors + records
$VENV -m streamlit run app.py                              # local validation viewer
```

## Status — what's proven vs pending
- **Offline:** every module + all E1–E6 run; **27 offline pytest green** plus Streamlit UI smoke.
- **Synthetic viewer:** the paired Exp-Force fixture has one strict winner per object, a 129-object
  experience pool with leave-one-out evaluation, detailed top-7 retrieval traces, cache telemetry,
  and persisted JSON/CSV benchmark results. It validates plumbing, not physical performance.
- **Prior live POC:** full Gemini stack ran on provisional **Exp-Force** fixture data (descriptor + `gemini-embedding-2`
  asymmetric retrieval + structured force prediction + GroupKFold). 6-object POC **MAE 0.083 N**
  (tiny synthetic/provisional sample; do not treat as a scientific result).
- **Tested live:** semantic retrieval + mass similarity + VLM force. **NOT yet live:**
  roughness/contact/physics/gripper-selection — the Exp-Force set has no roughness/contact labels
  (held constant), so those are validated **offline on synthetic data** and await our own
  2-gripper collection.
- **Free-tier quota** (`429 RESOURCE_EXHAUSTED`, ~20 requests) caps batch size; the disk cache
  makes re-runs resume for free. Space runs out or use a paid tier for the full 129 objects.

## Outstanding work / TODO
- Collect the real 2-gripper gecko/silicone dataset (`docs/data_collection_sop.md`); fill in
  gripper firmware pins/gains (`firmware/gripper_force`).
- Plots: parity (pred vs true) + k-sweep from the cached Exp-Force run; physical-relationship
  plots (force vs mass/roughness/contact + crossover map) — a sub-agent was drafting these under
  `docs/relationships/` (may or may not have landed; check).
- Integrate the sibling `../material-segmenter/` (Gemini segmentation + rembg + Marigold) as the
  descriptor / contact-fraction source (see `docs/backlog.md`).
- Full future-work list: `docs/backlog.md`.

## Reference
`docs/experiments.md` (E1–E6 protocol), `docs/backlog.md`, `docs/data_collection_sop.md`,
`docs/project-context.md` (original design rationale), `README.md`.
