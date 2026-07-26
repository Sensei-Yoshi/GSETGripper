# Streamlit Application Architecture

The research lab remains available through the original command:

```bash
streamlit run app.py
```

`app.py` is intentionally only a compatibility entrypoint. The application code lives in
the import-safe `streamlit_app` package; do not create a top-level package named `streamlit`,
because that can shadow the installed framework.

## Application flow

Each Streamlit rerun follows one path:

1. `app.py` calls `streamlit_app.app.main()`.
2. The shell calls `st.set_page_config`, applies the unchanged global CSS, and imports the
   tab registry.
3. The shell discovers every non-hidden direct folder under `data/` except `cache`, renders
   the global **Dataset** selector, and calls `load_context()` for the selected folder.
4. `load_context()` creates a dataset-scoped runtime configuration and one immutable
   `AppContext` containing the catalog, active `Dataset`, object rows, and summary.
5. The shell creates tabs from `TAB_SPECS`, in registry order, and calls each renderer with
   the same context.

Streamlit executes the body of every `st.tabs` container on each rerun, including tabs that
are not selected. A tab renderer must therefore avoid expensive or side-effectful work unless
the user activates a button or another explicit control. Cache decorators and lazy imports
used by the camera and background-removal model preserve this rule.

## Module responsibilities

| Module | Responsibility |
|---|---|
| `app.py` | Stable command-line entrypoint; contains no UI implementation. |
| `streamlit_app/app.py` | Page configuration, global frame, context loading, and tab dispatch. |
| `streamlit_app/context.py` | `AppContext` and the single context-loading function. |
| `streamlit_app/style.py` | The page-wide CSS. Only the application shell applies it. |
| `streamlit_app/prediction_ui.py` | Shared prediction metrics, truth display, retrieval table, formula, and formatting. |
| `streamlit_app/tabs/registry.py` | Ordered `TabSpec` declarations and the tab-renderer contract. |
| `streamlit_app/tabs/*.py` | One renderer per visible top-level tab, plus private tab-specific helpers. |
| `modules/datasets/` | Folder discovery, dataset/object contracts, artifact loading, and stage-selectable preparation. |
| `modules/cache.py` | Dataset-scoped JSON cache with read-through for the legacy flat Exp-Force cache. |
| `modules/models/` | Lazy Gemini, background-removal, and Marigold adapters. |

The **Runs Viewer** owns saved single runs, saved benchmarks, and resumable E1–E4 suite
comparison/export. The **Data Viewer** is intentionally limited to the dataset and
descriptor catalog. The **Prompts & Embodiments** tab edits `prompts.yaml` atomically,
including the fixed written descriptions of both grippers.

The **Marigold Roughness** tab is intentionally independent of the force pipeline. It can
run the active dataset's images or an uploaded override, and it browses all prior runs under
`test_data/marigold_tests`. The default background-removal pass saves a mask and transparent
cutout and restricts roughness statistics to the foreground. Heavy rembg/Torch/Diffusers
imports and model loading occur only after the user presses the run button; cached model
adapters survive ordinary Streamlit reruns.

Every top-level tab exports exactly this entrypoint:

```python
def render(context: AppContext) -> None:
    ...
```

Use `context.config`, `context.dataset`, `context.rows`, and `context.summary` instead of
independently calling `load_config`, `load_rows`, or `validation_summary`. This keeps all tabs
on the same active-dataset snapshot during a rerun. Dataset-dependent output paths must come
from `context.dataset.paths` or the runtime `context.config`, never a fixed `data/expforce`
constant.

## Dataset contract and global selection

The global selector is rendered once in `streamlit_app/app.py`, above the tab containers.
Changing it clears transient run/preparation/contact results, and every tab receives the new
context on the same rerun. Empty folders are valid catalog entries; they show zero objects.

`Dataset` is the aggregate. Its `objects` mapping contains `DatasetObject` values with stable
attributes for `image`, `description`, `embedding`, optional `mass_g`, optional
`roughness_class`, optional `projected_contact_fraction`, and optional paired
`gripper_outcomes`. Description and embedding values are artifact objects with provenance,
not loose parallel dictionaries. Convenience mappings (`dataset.images`,
`dataset.descriptions`, and `dataset.embeddings`) are available for analysis.

The adapter selected from folder contents determines capabilities:

- `dataset_2gripper.csv` uses the paired CSV adapter and enables experiences, Single Run,
  benchmarks, and suites when its required measurements and labels validate.
- other folders use the image-folder adapter. Source images may be at the folder root or in
  nested folders. Generated masks, cutouts, contact plots, descriptors, runs, results, and
  other artifact folders are excluded from the source inventory.

The Data Preparation tab exposes independent **Gemini descriptions**, **Text embeddings**,
and **Experience records** checkboxes. Indexing is always implicit. Embeddings and experience
records add descriptions as a prerequisite, but choosing descriptions alone does not run
either downstream stage. Checkpoints and the manifest live inside the active dataset folder.
The equivalent CLI is `scripts/prepare_dataset.py`.

## Adding a tab

These steps are the extension contract for either a human or an AI coding agent:

1. Create `streamlit_app/tabs/<tab_name>.py` with a public
   `render(context: AppContext) -> None` function. Keep helper functions private to that module
   unless a second tab genuinely needs them.
2. Import the module in `streamlit_app/tabs/registry.py` and add one `TabSpec` at the desired
   position in `TAB_SPECS`. The tuple order is the visible tab order.
3. Give interactive widgets explicit, globally unique `key` values whenever labels may be
   repeated in another tab. Because all tabs render on every rerun, keys are global across the
   application rather than local to a tab.
4. Reuse the shared context and prediction helpers described below. Do not call
   `st.set_page_config`, apply global CSS, mutate `TAB_SPECS`, or render UI at import time.
5. Add the new label to `EXPECTED_TABS` in `tests/test_streamlit_app.py` and add focused tests
   for any nontrivial behavior. Run the Streamlit smoke test and the normal project checks.

Do not turn the app into Streamlit's filesystem-based multipage navigation unless that is an
explicit product decision; the current interface is one page with ordered `st.tabs`.

## Shared prediction helpers

`streamlit_app.prediction_ui` owns UI behavior shared by Single Run and the Runs Viewer
saved-run inspector:

- `format_force` and `format_experiment` preserve display formatting.
- `truth_for_display` and `truth_payload` produce the validation labels used by the UI and
  saved run artifacts.
- `paired_retrieval_table` builds the full paired-neighbor table.
- `render_formula` draws the hybrid-similarity formula with normalized weights.
- `render_prediction` draws the complete result area, including metrics, counterfactual or
  truth status, per-gripper predictions, physics trace, retrieval evidence, and formula.

New tabs displaying a `PipelineRunResult` should call `render_prediction` rather than copying
its layout. Pass the exact `Config` used for that result so retrieval counts and offline/live
captions remain accurate. For an unscored result, set `counterfactual=True`; `truth` may only be
absent on that branch.

## Behavior that must remain stable

- `app.py` remains runnable from the project root.
- Only the shell calls `st.set_page_config`, and it does so before visible UI rendering.
- The Dataset selector remains above the tabs and applies to every dataset-dependent tab.
- Existing tab order, widget keys, and stored payload shapes are compatibility-sensitive.
- Session-state keys and stored payload shapes are shared across Single Run, Runs Viewer,
  and Cache Status; changes require coordinated migration and tests.
- Contact Fraction keeps its cached camera/model resources and lazy optional-dependency
  imports; outputs live under `data/<active-dataset>/contact_fraction`.
- Marigold Roughness keeps optional dependencies lazy and writes only to
  `test_data/marigold_tests/<run-id>/`.
- Tab registration must not change any public interface in `modules`.

Run the UI regression test after any tab or shared-component change:

```bash
../../env/bin/python -m pytest tests/test_streamlit_app.py
../../env/bin/python -m ruff check .
../../env/bin/python -m streamlit run app.py --server.headless true
```
