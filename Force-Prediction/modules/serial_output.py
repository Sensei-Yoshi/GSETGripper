"""Optional serial delivery of one selected gripper force prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .contracts import Gripper

BAUD_RATE = 9600
DEFAULT_TIMEOUT_SECONDS = 30.0

_SELECT_TOKENS = {
    Gripper.GECKO: "GEKKO",  # Firmware spelling in arduino/main/main.ino.
    Gripper.SILICONE: "SILICONE",
}


@dataclass(frozen=True)
class SerialPort:
    device: str
    description: str
    manufacturer: str | None = None

    @property
    def label(self) -> str:
        details = self.description
        if self.manufacturer and self.manufacturer.lower() not in details.lower():
            details = f"{details} · {self.manufacturer}"
        return f"{self.device} — {details}" if details else self.device


@dataclass(frozen=True)
class SerialSendResult:
    port: str
    gripper: Gripper
    force_n: float
    warnings: tuple[str, ...] = ()
    boot_messages: tuple[str, ...] = ()


def list_serial_ports() -> list[SerialPort]:
    """Return currently connected serial ports without opening any of them."""
    from serial.tools import list_ports

    return [
        SerialPort(
            device=port.device,
            description=port.description or "",
            manufacturer=port.manufacturer,
        )
        for port in sorted(list_ports.comports(), key=lambda item: item.device)
    ]


def _read_reply(connection, context: str) -> str:  # noqa: ANN001
    reply = connection.readline().decode("ascii", errors="ignore").strip()
    if not reply:
        raise TimeoutError(f"no serial reply while waiting for {context}")
    return reply


def _wait_until_ready(connection) -> list[str]:  # noqa: ANN001
    messages: list[str] = []
    while True:
        reply = _read_reply(connection, "firmware READY")
        if reply == "READY":
            return messages
        if reply.startswith("ERR"):
            raise RuntimeError(f"firmware boot failed: {reply}")
        if reply.startswith("INFO") or reply.startswith("TARING"):
            messages.append(reply)
            continue
        raise RuntimeError(f"unexpected firmware boot reply: {reply}")


def _send_command(connection, line: str, expected: str) -> list[str]:  # noqa: ANN001
    connection.write((line + "\n").encode("ascii"))
    connection.flush()
    warnings: list[str] = []
    while True:
        reply = _read_reply(connection, expected)
        if reply.startswith("ERR"):
            raise RuntimeError(f"firmware rejected {line!r}: {reply}")
        if reply.startswith("WARN"):
            warnings.append(reply)
            continue
        if reply == expected:
            return warnings
        if reply.startswith("DONE"):
            raise RuntimeError(
                f"firmware desynchronised: expected {expected!r} "
                f"for {line!r}, got {reply!r}"
            )
        raise RuntimeError(f"unexpected firmware reply to {line!r}: {reply}")


def _force_text(force_n: float) -> str:
    # main.ino's strict parser rejects scientific notation. Six fractional
    # digits preserve small predictions without sending an exponent.
    text = f"{force_n:.6f}".rstrip("0").rstrip(".")
    if text == "0":
        raise ValueError("force rounds to zero at serial command precision")
    return text


def send_force(
    port: str,
    gripper: Gripper | str,
    force_n: float | None,
    limit_n: float,
    *,
    baud: int = BAUD_RATE,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> SerialSendResult:
    """Select a gripper and execute the firmware's force-seek command once."""
    if not port.strip():
        raise ValueError("serial port is required")
    selected_gripper = Gripper(gripper)
    if force_n is None:
        raise ValueError("cannot send a missing force")
    force = float(force_n)
    if not math.isfinite(force) or force <= 0:
        raise ValueError("serial force must be finite and greater than zero")
    if not math.isfinite(limit_n) or limit_n <= 0:
        raise ValueError("serial force limit must be finite and greater than zero")
    if force > limit_n:
        raise ValueError(f"force {force:g} N exceeds serial safety limit {limit_n:g} N")
    force_text = _force_text(force)

    import serial

    connection = serial.Serial(port, baud, timeout=timeout)
    try:
        boot_messages = _wait_until_ready(connection)
        warnings = _send_command(
            connection,
            f"SELECT {_SELECT_TOKENS[selected_gripper]}",
            "DONE SELECT",
        )
        warnings.extend(
            _send_command(connection, f"FORCE {force_text}", "DONE FORCE")
        )
        return SerialSendResult(
            port=port,
            gripper=selected_gripper,
            force_n=force,
            warnings=tuple(warnings),
            boot_messages=tuple(boot_messages),
        )
    finally:
        connection.close()
