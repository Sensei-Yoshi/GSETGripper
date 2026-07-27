"""Global test safety guards."""

from __future__ import annotations

import pytest

from modules.models.gemini import GeminiClient


@pytest.fixture(autouse=True)
def block_unapproved_gemini_network(monkeypatch, request):  # noqa: ANN001, ANN201
    if request.node.get_closest_marker("gemini_integration"):
        return

    def blocked(_self):  # noqa: ANN001, ANN202
        raise AssertionError(
            "unit test attempted real Gemini access; install an explicit test fake"
        )

    monkeypatch.setattr(GeminiClient, "_sdk", blocked)
