"""Usage instructions and experiment reference tab."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from modules.config import EXPERIMENT_IDS
from modules.experiments import EXPERIMENT_CATALOG
from streamlit_app.context import AppContext


def render(context: AppContext) -> None:
    base_cfg = context.config
    dataset = context.dataset
    object_count = len(dataset.objects)
    st.header("How to use the lab")
    st.markdown(
        f"""
1. Choose a dataset in the global selector above. Every dataset-aware tab and artifact path
   follows that selection.
2. Open **Data Preparation** and select only the stages you need. Use **Live Gemini** when
   you want image-derived descriptions or Gemini text embeddings.
3. Open **Single Run**, choose one of the {object_count} active objects, or upload an image.
   This tab is available only when the dataset has measurements and paired force labels.
4. For conditions that use sensors, set mass, roughness, and projected contact fraction.
   E1 and E3 intentionally disable these controls. Changing an enabled value or image
   creates an unscored counterfactual that uses the full experience pool.
5. Choose an experiment and **Offline** or **Live Gemini**. Adjust retrieval weights and
   hybrid similarity constants for E4; E3 is fixed to semantic cosine similarity.
6. Click **Run pipeline**. Read the selected gripper and force first, then compare both
   predictions, evidence summaries, and the top-five matches.
7. Use **Benchmark** for one condition or **Runs Viewer** for a resumable E1–E4
   comparison and saved-run inspection. Use **Data Viewer** for the experience catalog and
   **Cache Status** to confirm that repeated Gemini requests are being reused.
        """
    )

    st.header("Experiment definitions")
    uses = {
        "e1": "Image + joint VLM",
        "e2": "Image + sensors + joint VLM",
        "e3": "Image + semantic-only paired retrieval + joint VLM",
        "e4": "Sensors + paired retrieval + joint VLM",
        "e5": "Sensors + calibrated physics",
        "e6": "Sensors + calibrated physics + semantic residual",
    }
    experiments = pd.DataFrame(
        [
            {
                "Experiment": experiment_id.upper(),
                "Uses": uses[experiment_id],
                "Meaning": EXPERIMENT_CATALOG[experiment_id].summary,
            }
            for experiment_id in EXPERIMENT_IDS
        ]
    )
    st.dataframe(experiments, hide_index=True, width="stretch")
    st.caption(
        "Primary comparisons: E1 vs E2 tests sensors without experience; E1 vs E3 tests "
        "semantic experience without sensors; E4 tests their combined system. E5 vs E6 "
        "isolates the learned residual."
    )

    st.subheader("Prediction prompt routing")
    st.caption(
        "Only VLM force-prediction experiments have prompts. All text below is loaded "
        "directly from prompts.yaml."
    )
    with st.expander("Shared prediction system prompt"):
        st.code(base_cfg.prompts.prediction_system, language=None)
    for experiment in ("e1", "e2", "e3", "e4"):
        prompt_key = base_cfg.experiment(experiment).prompt
        assert prompt_key is not None
        with st.expander(f"{experiment.upper()} instruction — prompts.experiments.{prompt_key}"):
            st.code(base_cfg.prompts.experiments[prompt_key], language=None)
    st.info("E5 and E6 do not call a VLM for force prediction, so they have no prompt.")

    st.subheader("How E6 learns the physics residual")
    st.markdown(
        """
E6 calibrates the same physics model as E5 on each training fold, then learns the target
`measured force − physics force` separately for gecko and silicone. The default
gradient-boosted trees use log mass, roughness, projected contact fraction, physics
force, and PCA-reduced semantic embedding features. At inference the continuous output
is `physics force + predicted residual`, clamped only to the 0–8 N hardware range.
E6 retrieves no neighbors and makes no VLM force-prediction call.
        """
    )

    st.header("What E3/E4 + Live Gemini means")
    st.markdown(
        """
For a known dataset object, the pipeline excludes that object and uses the other objects.
For a custom query, it uses the full active dataset. It reuses one cached text embedding per reference object for
both grippers and embeds the query description. E3 ranks by semantic cosine only and hides
all sensor values. E4 ranks with the displayed semantic + sensor hybrid similarity. Each
retrieves five objects once, with both Gecko and silicone outcomes attached to each object.
One Gemini request returns both force estimates and an explicit recommendation. Neither
condition constructs or sends a physics estimate. Python selects the lower feasible
predicted force and explicitly reports a prediction tie only when the continuous estimates are equal.

**Live Gemini** means Gemini-backed descriptor, embedding, and VLM stages are allowed to make
network calls. A cached identical request makes no new call. It does not mean the labels are
real, and it does not automatically replace an offline-prepared reference corpus with visual
descriptions; that is the job of live Data Preparation.
        """
    )

    manifest_path = dataset.paths.preparation_manifest
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        missing = len(manifest.get("missing_images", []))
        stages = manifest.get("stages", {})
        descriptors_done = manifest.get(
            "descriptors_completed", stages.get("descriptions", {}).get("completed", 0)
        )
        embeddings_done = manifest.get(
            "embeddings_completed", stages.get("embeddings", {}).get("completed", 0)
        )
        statuses = [stage.get("status") for stage in stages.values()]
        status = manifest.get("status") or (
            "failed" if "failed" in statuses else "complete"
        )
        if status == "complete" and missing == 0:
            st.success(
                f"Current experience pool: {descriptors_done} descriptors and "
                f"{embeddings_done} warmed reference text embeddings."
            )
        else:
            st.warning(
                f"Preparation status: {status}; {descriptors_done} "
                f"descriptors, {embeddings_done} embeddings, and {missing} unavailable images."
            )
    else:
        st.warning("No preparation manifest exists yet. Run Data Preparation before evaluation.")

    st.header("Application sections")
    sections = pd.DataFrame(
        [
            {
                "Section": "Single Run",
                "Purpose": "Inspect one leave-one-out or custom query with full evidence.",
            },
            {
                "Section": "Benchmark",
                "Purpose": "Evaluate every object using leave-one-object-out training.",
            },
            {
                "Section": "Runs Viewer",
                "Purpose": "Resume E1–E4 suites, compare/export results, and inspect saved runs.",
            },
            {
                "Section": "Data Viewer",
                "Purpose": "Browse active-dataset images, measurements, descriptions, and embedding status.",
            },
            {
                "Section": "Prompts & Embodiments",
                "Purpose": "Edit prompts and the fixed written descriptions of both grippers.",
            },
            {
                "Section": "Data Preparation",
                "Purpose": "Run selected, resumable description, embedding, or experience stages.",
            },
            {
                "Section": "Cache Status",
                "Purpose": "Inspect saved API responses and telemetry from the latest single run.",
            },
            {
                "Section": "Help & Experiments",
                "Purpose": "Understand controls, experiment ablations, and result interpretation.",
            },
        ]
    )
    st.dataframe(sections, hide_index=True, width="stretch")
    st.info(
        "All forces, winner labels, and reported accuracy are synthetic pipeline-validation "
        "signals. They can reveal software or modeling behavior, but cannot establish physical "
        "gripper performance."
    )
