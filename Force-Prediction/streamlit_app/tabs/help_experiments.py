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
2. Open **Data Preparation** and select only the stages you need. Descriptions and semantic
   embeddings use Gemini and reuse exact cached requests.
3. Open **Single Run**, choose one of the {object_count} active objects, or upload an image.
   E1 can run from an image alone; other conditions report exactly which inputs or references
   are still missing.
4. Select Gecko, silicone, or both with the global prediction checkboxes. Record every
   measurement required by the selected fixed experiment profile.
5. Choose an experiment. Adjust retrieval weights for E4-E6; E3 is fixed to semantic cosine.
6. Click **Run pipeline**. Single-target runs show force and feasibility; paired runs also
   compare candidates and select the lowest-force feasible gripper.
7. Use **Benchmark** to save truth-free predictions and evaluate them later without model
   calls. Use **Runs Viewer** for versioned results, a resumable E1–E6 comparison, and
   saved-run inspection. Use **Data Viewer** for the experience catalog and
   **Cache Status** to confirm that repeated Gemini requests are being reused.
        """
    )

    st.header("Experiment definitions")
    uses = {
        "e1": "Image + target-aware VLM",
        "e2": "Image + sensors + target-aware VLM",
        "e3": "Image + semantic-only retrieval + target-aware VLM",
        "e4": "Semantic + mass retrieval + target-aware VLM",
        "e5": "E4 + continuous roughness",
        "e6": "E5 + projected contact",
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
        "semantic experience without sensors; E4-E6 isolate the incremental value of "
        "mass, roughness, and projected contact."
    )

    st.subheader("Prediction prompt routing")
    st.caption(
        "Only VLM force-prediction experiments have prompts. All text below is loaded "
        "directly from prompts.yaml."
    )
    with st.expander("Shared prediction system prompt"):
        st.code(base_cfg.prompts.prediction_system, language=None)
    for experiment in EXPERIMENT_IDS:
        prompt_key = base_cfg.experiment(experiment).prompt
        assert prompt_key is not None
        with st.expander(f"{experiment.upper()} instruction — prompts.experiments.{prompt_key}"):
            st.code(base_cfg.prompts.experiments[prompt_key], language=None)

    st.header("What Gemini-backed E3-E6 means")
    st.markdown(
        """
For a known dataset object, the pipeline excludes that object and uses eligible outcomes from the other objects.
For a custom query, it uses the eligible outcomes in the active dataset. It reuses one cached text embedding per reference object for
the active grippers and embeds the query description. E3 ranks by semantic cosine only and hides
all sensor values. E4 adds mass, E5 adds continuous roughness, and E6 adds projected contact.
The hybrid conditions receive the normalized ranking weights as provenance; those weights are
not a force equation. Each
retrieves five objects once, with only active-gripper outcomes attached to each object.
One target uses a per-gripper response; two targets use one joint response. Neither
condition constructs or sends a physics estimate. Python selects the lower feasible
predicted force and explicitly reports a prediction tie only when the continuous estimates are equal.

Gemini-backed descriptor, embedding, and VLM stages may make network calls. A cached identical
request makes no new call. This does not mean the labels are real, and pipeline execution does
not automatically replace a previously prepared reference corpus; run Data Preparation to
refresh visual descriptions and reference embeddings.
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
                "Purpose": "Generate immutable predictions, then evaluate available truth separately.",
            },
            {
                "Section": "Runs Viewer",
                "Purpose": "Inspect prediction/evaluation history and run two-stage E1–E6 suites.",
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
