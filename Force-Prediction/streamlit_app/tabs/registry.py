"""Ordered registry of top-level Streamlit tabs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from streamlit_app.context import AppContext
from streamlit_app.tabs import (
    benchmark,
    cache_status,
    contact_fraction,
    data_preparation,
    data_viewer,
    help_experiments,
    prompts_editor,
    roughness_prediction,
    runs_viewer,
    single_run,
)

TabRenderer = Callable[[AppContext], None]


@dataclass(frozen=True)
class TabSpec:
    """A visible tab label and the function that renders its contents."""

    label: str
    render: TabRenderer


TAB_SPECS: tuple[TabSpec, ...] = (
    TabSpec("Single Run", single_run.render),
    TabSpec("Benchmark", benchmark.render),
    TabSpec("Runs Viewer", runs_viewer.render),
    TabSpec("Data Viewer", data_viewer.render),
    TabSpec("Prompts & Embodiments", prompts_editor.render),
    TabSpec("Contact Fraction", contact_fraction.render),
    TabSpec("Marigold Roughness", roughness_prediction.render),
    TabSpec("Data Preparation", data_preparation.render),
    TabSpec("Cache Status", cache_status.render),
    TabSpec("Help & Experiments", help_experiments.render),
)
