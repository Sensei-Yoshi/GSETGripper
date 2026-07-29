"""Regression tests for the modular Streamlit application shell."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import streamlit
from streamlit.testing.v1 import AppTest

from streamlit_app.tabs.registry import TAB_SPECS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TABS = [
    "Single Run",
    "Benchmark",
    "Runs Viewer",
    "Data Viewer",
    "Prompts & Embodiments",
    "Contact Fraction",
    "Marigold Roughness",
    "Data Preparation",
    "Cache Status",
    "Help & Experiments",
]


def test_default_app_structure_matches_research_lab() -> None:
    app = AppTest.from_file(PROJECT_ROOT / "app.py").run(timeout=30)

    assert list(app.exception) == []
    assert [item.value for item in app.title] == ["Force Pipeline Lab"]
    assert [item.label for item in app.tabs] == EXPECTED_TABS
    button_labels = [item.label for item in app.button]
    for label in (
        "Run pipeline",
        "Run selected",
        "Run/Resume suite predictions",
        "Evaluate suite & generate comparison",
        "Validate prompt bundle",
        "Save prompts & embodiments",
        "Capture & Analyze",
        "Run Marigold",
        "Run selected preparation stages",
        "+ Add condition",
    ):
        assert label in button_labels
    selectbox_labels = [item.label for item in app.selectbox]
    for label in (
        "Dataset",
        "Dataset object",
        "Experiment profile",
        "Experiment",
        "Dataset split",
        "Saved suite",
    ):
        assert label in selectbox_labels
    multiselect_labels = {item.label for item in app.multiselect}
    assert {"Marigold dataset images", "Marigold analyses"} <= multiselect_labels
    marigold_uploader = next(
        item
        for item in app.file_uploader
        if item.label == "Upload one or more additional images"
    )
    assert marigold_uploader.accept_multiple_files is True
    checkbox_labels = {item.label for item in app.checkbox}
    assert {
        "Gemini descriptions",
        "Text embeddings",
        "Marigold roughness",
        "Surface/contact fraction from image_2",
        "Experience records",
        "Consider measured roughness",
        "Consider projected contact fraction",
        "Predict Gecko force",
        "Predict silicone force",
    } <= checkbox_labels
    markdown_values = {item.value for item in app.markdown}
    assert "**Image 1 — Perception view**" in markdown_values
    assert "**Image 2 — Geometry view**" in markdown_values
    assert not any(
        "Each card combines CSV source fields" in item.value
        for item in app.info
    )
    expected_state = {
        "active_dataset_id",
        "consider_roughness",
        "consider_projected_contact",
        "predict_gecko_force",
        "predict_silicone_force",
        "benchmark_experiment",
        "run_benchmark_predictions",
        "catalog_page",
        "catalog_page_size",
        "runs_viewer_mode",
        "suite_cost_confirmation",
        "suite_selector",
        "run_primary_suite",
        "evaluate_primary_suite",
        "validate_prompt_bundle",
        "save_prompt_bundle",
        "roughness_view_history",
        "roughness_dataset_images",
        "roughness_override_images",
        "marigold_analysis_modes",
        "roughness_processing_resolution",
        "roughness_inference_steps",
        "roughness_ensemble_size",
        "roughness_seed",
        "roughness_remove_background",
        "roughness_crop_padding_ratio",
        "roughness_contact_band_fraction",
        "roughness_mask_erosion_ratio",
        "topography_base_surface_sigma_ratio",
        "run_marigold",
        "prompt_prediction_system",
        "prompt_descriptor_system",
        "prompt_descriptor_instruction",
        "prompt_instruction_e1",
        "prompt_instruction_e2",
        "prompt_instruction_e3",
        "prompt_instruction_e4",
        "prompt_target_single",
        "prompt_target_joint",
        "embodiment_description_gecko",
        "embodiment_description_silicone",
        "prepare_descriptions",
        "prepare_embeddings",
        "prepare_roughness",
        "prepare_surface_area",
        "prepare_experiences",
        "run_preparation_stages",
    }
    assert expected_state <= set(app.session_state.filtered_state)
    assert any("Benchmark split · Train:" in item.value for item in app.caption)


def test_global_gripper_controls_require_at_least_one_target() -> None:
    app = AppTest.from_file(PROJECT_ROOT / "app.py").run(timeout=30)
    gecko = next(item for item in app.checkbox if item.label == "Predict Gecko force")
    silicone = next(
        item for item in app.checkbox if item.label == "Predict silicone force"
    )

    assert gecko.value is True and silicone.value is True
    gecko.set_value(False)
    app.run(timeout=30)

    silicone = next(
        item for item in app.checkbox if item.label == "Predict silicone force"
    )
    assert app.session_state["predict_gecko_force"] is False
    assert app.session_state["predict_silicone_force"] is True
    assert silicone.disabled is True


def test_descriptor_editor_shows_the_standard_retrieval_template() -> None:
    app = AppTest.from_file(PROJECT_ROOT / "app.py").run(timeout=30)

    descriptor = next(
        item for item in app.text_area if item.label == "Descriptor instruction"
    )
    assert (
        "[Material/object form]. The intended contact region is [surface condition], "
        "[geometry], and [other relevant properties]."
    ) in descriptor.value


def test_marigold_tab_accepts_multiple_dataset_images() -> None:
    app = AppTest.from_file(PROJECT_ROOT / "app.py").run(timeout=30)
    picker = next(
        item for item in app.multiselect if item.label == "Marigold dataset images"
    )
    selected = picker.options[:2]

    picker.set_value(selected)
    app.run(timeout=30)

    assert list(app.exception) == []
    assert app.session_state["roughness_dataset_images"] == selected
    assert any(item.value == "2 images selected" for item in app.caption)


def test_tab_registry_has_unique_labels_and_common_renderer_contract() -> None:
    labels = [spec.label for spec in TAB_SPECS]

    assert labels == EXPECTED_TABS
    assert len(labels) == len(set(labels))
    for spec in TAB_SPECS:
        assert callable(spec.render)
        assert list(inspect.signature(spec.render).parameters) == ["context"]


def test_streamlit_import_resolves_external_framework() -> None:
    streamlit_path = Path(streamlit.__file__).resolve()

    assert PROJECT_ROOT not in streamlit_path.parents
    assert streamlit_path.name == "__init__.py"


def test_global_dataset_selector_switches_every_tab_to_image_only_dataset() -> None:
    app = AppTest.from_file(PROJECT_ROOT / "app.py").run(timeout=30)

    image_only_dataset = None
    for dataset_id in app.selectbox[0].options:
        if dataset_id == "expforce":
            continue
        app.selectbox[0].set_value(dataset_id)
        app.run(timeout=30)
        if any("objects · image folder" in item.value for item in app.caption):
            image_only_dataset = dataset_id
            break
    if image_only_dataset is None:
        pytest.skip("workspace does not currently contain an image-folder dataset")

    assert list(app.exception) == []
    assert app.session_state["active_dataset_id"] == image_only_dataset
    assert any("objects · image folder" in item.value for item in app.caption)
    experiment = next(item for item in app.selectbox if item.label == "Experiment profile")
    experiment.set_value("e1")
    app.run(timeout=30)
    run = next(item for item in app.button if item.label == "Run pipeline")
    assert not run.disabled
    assert any(item.label == "Mass (g)" for item in app.number_input)
    assert any(item.label == "Status" for item in app.selectbox)
    experience_control = next(
        item for item in app.checkbox if item.label == "Experience records"
    )
    assert experience_control.disabled
    surface_control = next(
        item
        for item in app.checkbox
        if item.label == "Surface/contact fraction from image_2"
    )
    assert surface_control.disabled
