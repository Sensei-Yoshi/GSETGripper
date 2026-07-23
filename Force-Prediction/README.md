# Material-Aware Gripper Selection & Minimum-Force Prediction

Given an unseen object, jointly (a) choose **TPU–gecko** vs **TPU–silicone** and
(b) predict the minimum stationary-finger normal force `F*(o, g)` to lift it.
Force is treated as an *object–gripper interaction* property with a material
crossover — the system sits between **Exp-Force** (experience-conditioned VLM,
no physics) and **DeliGrasp** (LLM-inferred physics, no experience):

> Measure what can be measured → hybrid RAG retrieves paired objects once with
> both force labels → one Gemini response predicts both forces → a deterministic
> selector takes the feasible arg-min. Calibrated physics is evaluated separately
> in E4 and E6 rather than being sent to the E5 VLM.

## Quickstart (fully offline — no hardware, no API key)

```bash
make setup            # editable install + dev tools
make test             # unit tests
make smoke            # mock dataset -> run all configured experiments offline
```

Stage-by-stage debugging (each runs standalone):

```bash
python scripts/check_hardware.py     # mock gripper + staircase
python scripts/check_perception.py   # descriptor + contact-fraction proxy
python scripts/check_retrieval.py    # hybrid retrieval + paired deltas
python scripts/check_physics.py      # calibration + min-force solver
python scripts/check_prediction.py   # one per-gripper VLM call (dry-run stub)
python scripts/check_pipeline.py     # full E5 on one object, every stage printed
```

## Exp-Force pipeline viewer

The local Streamlit viewer uses the synthetic `dataset_2gripper.csv`, whose paired
forces have one strict winning gripper per object. All 129 objects form the experience
pool. Known-object evaluation is leave-one-object-out; custom queries use all 129.
The viewer calls the same `Pipeline` used by the experiment runner. E5 retrieves
five paired objects once and uses one structured Gemini request to predict both
gripper forces:

```bash
pip install -e ".[viewer,gemini]"
python scripts/prepare_expforce_viewer.py
streamlit run app.py
```

Use `python scripts/prepare_expforce_viewer.py --live` to download all images,
checkpoint contact-region Gemini descriptions per object, and warm one text-only
reference embedding per object. Preparation is resumable after quota interruptions.
The UI defaults to offline E5; Live Gemini uses the content-addressed cache.
The **Data Viewer** browses all 129 images/descriptions and exact saved single runs.
The **Help & Experiments** tab explains the lab workflow and every experiment,
what a live E5 run executes, and the difference between preparation and evaluation.

## Live runs

Set a key and flip `models.dry_run: false` in `config.yaml`:

```bash
export GEMINI_API_KEY=...          # or GOOGLE_API_KEY
pip install -e ".[gemini]"
python scripts/run_experiment.py --exp e5
```

## Data collection

```bash
python -m force_prediction.collect --mock --n 40          # synthetic bench
python -m force_prediction.collect --port /dev/cu.usbmodemXXXX   # real hardware
```

Real collection needs the gripper firmware (`firmware/gripper_force`), the LED
roughness system, a scale, and the Astra+ camera. See
[`docs/data_collection_sop.md`](docs/data_collection_sop.md).

## Where things live

| File | Responsibility |
|------|----------------|
| `config.yaml` | **Every tunable + all prompts + the E1..E6 experiment toggles.** Tune here, never in code. |
| `force_prediction/contracts.py` | Shared Pydantic models (records, paired object view, predictions). |
| `force_prediction/hardware.py` | Device interfaces + real serial drivers + physics-backed mocks. |
| `force_prediction/physics.py` | Reduced-order gecko/silicone models, solver, calibration. |
| `force_prediction/retrieval.py` | Gemini Embedding 2 + hybrid similarity + paired-object and branch baselines. |
| `force_prediction/perception.py` | Descriptor + depth height-ratio contact fraction. |
| `force_prediction/prediction.py` | Gemini estimator + non-VLM baselines + deterministic selector. |
| `force_prediction/learning.py` | Physics-residual model (E6). |
| `force_prediction/evaluation.py` | GroupKFold splits + force/selection/regret metrics. |
| `force_prediction/pipeline.py` | **One** toggle-driven orchestration shared by all experiments. |
| `force_prediction/collect.py` | Coarse-to-fine staircase GT controller. |
| `scripts/` | Manual stage checks + `run_experiment.py`. |
| `docs/` | Experiment protocol, backlog, collection SOP. |

## Experiments

Defined as toggle sets in `config.yaml`; run with `python scripts/run_experiment.py --exp eN`.
See [`docs/experiments.md`](docs/experiments.md).

| E1 | E2 | E3 | E3b | E4 | E5 | E6 |
|----|----|----|-----|----|----|----|
| vision-only | +measured | +retrieval | retrieval-only (no VLM) | physics-only (no VLM) | **one paired retrieval + one joint VLM call** | physics+residual |

## Team workstreams (2–3 week sprint, 6 people)

Phase 0 (all): land `config.yaml`, `contracts.py`, `hardware.py` mocks, `pipeline.py`, CI.
Then parallel: **B** firmware+collection (start GT ASAP) · **C** perception ·
**D** retrieval · **E** physics+residual · **F** prediction+eval+runner · **A** contracts→eval.
The mock bench keeps every ML workstream unblocked while hardware is built.
