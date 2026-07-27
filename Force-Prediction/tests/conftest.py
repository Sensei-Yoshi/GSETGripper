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


@pytest.fixture(autouse=True)
def block_real_serial_ports(monkeypatch):  # noqa: ANN001, ANN201
    """Mirror of the Gemini guard above, for hardware.

    A unit test that opened a real port could drive an unhomed rig into the
    table -- there is no limit switch to stop it (see main.ino's Z-geometry
    comment block).
    """
    import serial

    def blocked(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError(
            "unit test attempted to open a real serial port; install a test fake"
        )

    monkeypatch.setattr(serial, "Serial", blocked)
