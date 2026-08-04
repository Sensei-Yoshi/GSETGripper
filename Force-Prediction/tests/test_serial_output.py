from __future__ import annotations

from types import SimpleNamespace

import pytest
import serial

from modules.contracts import Gripper
from modules.serial_output import list_serial_ports, send_force


class _FakeSerial:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.written: list[str] = []
        self.closed = False

    def readline(self) -> bytes:
        if not self.replies:
            return b""
        return (self.replies.pop(0) + "\n").encode("ascii")

    def write(self, payload: bytes) -> None:
        self.written.append(payload.decode("ascii"))

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def _install_connection(monkeypatch, replies: list[str]) -> _FakeSerial:  # noqa: ANN001
    connection = _FakeSerial(replies)
    monkeypatch.setattr(serial, "Serial", lambda *_args, **_kwargs: connection)
    return connection


def test_lists_ports_without_opening_them(monkeypatch):
    ports = [
        SimpleNamespace(device="/dev/cu.b", description="USB Serial", manufacturer="Arduino"),
        SimpleNamespace(device="/dev/cu.a", description="Gripper", manufacturer=None),
    ]
    monkeypatch.setattr("serial.tools.list_ports.comports", lambda: ports)

    found = list_serial_ports()

    assert [port.device for port in found] == ["/dev/cu.a", "/dev/cu.b"]
    assert found[1].label == "/dev/cu.b — USB Serial · Arduino"


@pytest.mark.parametrize(
    ("gripper", "select_line"),
    ((Gripper.GECKO, "SELECT GEKKO\n"), (Gripper.SILICONE, "SELECT SILICONE\n")),
)
def test_sends_selected_gripper_then_fixed_point_force(
    monkeypatch, gripper, select_line
):  # noqa: ANN001
    connection = _install_connection(
        monkeypatch,
        ["INFO cell 1 ready", "READY", "DONE SELECT", "DONE FORCE"],
    )

    result = send_force("/dev/fake", gripper, 1.75, 8.0)

    assert connection.written == [select_line, "FORCE 1.75\n"]
    assert connection.closed
    assert result.force_n == 1.75
    assert result.boot_messages == ("INFO cell 1 ready",)


def test_collects_firmware_warnings(monkeypatch):
    _install_connection(
        monkeypatch,
        ["READY", "WARN SELECT clamped", "DONE SELECT", "WARN force not reached", "DONE FORCE"],
    )

    result = send_force("/dev/fake", "gecko", 0.00001, 8.0)

    assert result.warnings == ("WARN SELECT clamped", "WARN force not reached")


@pytest.mark.parametrize("force", (None, 0.0, -1.0, float("nan"), float("inf"), 8.1))
def test_rejects_unsafe_force_before_opening_a_port(force):
    with pytest.raises(ValueError):
        send_force("/dev/fake", "gecko", force, 8.0)


def test_boot_error_closes_connection_without_sending(monkeypatch):
    connection = _install_connection(monkeypatch, ["ERR cell 1 timed out", "READY"])

    with pytest.raises(RuntimeError, match="boot failed"):
        send_force("/dev/fake", "gecko", 1.0, 8.0)

    assert connection.written == []
    assert connection.closed


def test_timeout_closes_connection_and_does_not_retry(monkeypatch):
    connection = _install_connection(monkeypatch, ["READY", "DONE SELECT"])

    with pytest.raises(TimeoutError, match="DONE FORCE"):
        send_force("/dev/fake", "gecko", 1.0, 8.0)

    assert connection.written == ["SELECT GEKKO\n", "FORCE 1\n"]
    assert connection.closed


def test_mismatched_ack_fails_without_sending_force(monkeypatch):
    connection = _install_connection(monkeypatch, ["READY", "DONE Z"])

    with pytest.raises(RuntimeError, match="desynchronised"):
        send_force("/dev/fake", "silicone", 1.0, 8.0)

    assert connection.written == ["SELECT SILICONE\n"]
    assert connection.closed
