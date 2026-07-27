from __future__ import annotations

import pytest

from modules.contracts import Gripper
from modules.handoff import GraspCommand
from modules.hardware import SerialGraspSender

COMMAND = GraspCommand(
    object_id="banana",
    object_height_mm=84.2,
    object_width_mm=61.0,
    gripper=Gripper.SILICONE,
    force_n=1.75,
)


class _FakeSerial:
    """Replays scripted firmware replies. An exhausted script models a timeout."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.written: list[str] = []

    def write(self, payload: bytes) -> None:
        self.written.append(payload.decode("ascii"))

    def flush(self) -> None:
        pass

    def readline(self) -> bytes:
        if not self.replies:
            return b""
        return (self.replies.pop(0) + "\n").encode("ascii")

    def close(self) -> None:
        pass


def _sender(replies: list[str]) -> tuple[SerialGraspSender, _FakeSerial]:
    # Bypass __init__ so no port is opened and no reset delay is incurred.
    sender = SerialGraspSender.__new__(SerialGraspSender)
    conn = _FakeSerial(replies)
    sender.conn = conn
    return sender, conn


def test_writes_every_line_in_order_and_waits_for_each_ack():
    sender, conn = _sender(["DONE SELECT", "DONE Z", "DONE FORCE", "DONE GRIP"])
    warnings = sender.send(COMMAND)
    assert conn.written == [
        "SELECT SILICONE\n",
        "Z 84.2\n",
        "FORCE 1.75\n",
        "GRIP CLOSE 61.0\n",
    ]
    assert warnings == []


def test_error_raises_and_does_not_hang():
    # sendErr returns without setting a pending flag (main.ino's sendErr), so
    # no DONE ever follows. A reader that waits for one would block forever.
    sender, conn = _sender(["ERR bad value"])
    with pytest.raises(RuntimeError, match="ERR bad value"):
        sender.send(COMMAND)
    assert conn.written == ["SELECT SILICONE\n"]


def test_warning_is_collected_and_not_mistaken_for_the_ack():
    sender, _ = _sender(
        [
            "WARN SELECT clamped to 430",
            "DONE SELECT",
            "DONE Z",
            "DONE FORCE",
            "DONE GRIP",
        ]
    )
    warnings = sender.send(COMMAND)
    assert warnings == ["WARN SELECT clamped to 430"]


def test_silent_board_times_out():
    sender, _ = _sender([])
    with pytest.raises(TimeoutError, match="no reply"):
        sender.send(COMMAND)


def test_unexpected_reply_raises_rather_than_looping():
    sender, _ = _sender(["HELLO"])
    with pytest.raises(RuntimeError, match="unexpected firmware reply"):
        sender.send(COMMAND)


def test_mismatched_ack_raises_instead_of_accepting_wrong_axis():
    # The first line sent is SELECT. A DONE Z reply here means the stream is
    # desynchronised -- e.g. boot-time garbage left over from the reset that
    # opening the port just triggered. Accepting any "DONE"-prefixed line
    # would let this pass as success.
    sender, conn = _sender(["DONE Z"])
    with pytest.raises(RuntimeError, match=r"expected 'DONE SELECT'.*got 'DONE Z'"):
        sender.send(COMMAND)
    assert conn.written == ["SELECT SILICONE\n"]


def test_force_line_without_its_ack_times_out():
    # FORCE is verified like every other axis: if its "DONE FORCE" ack never
    # arrives (e.g. the board resets mid-sequence), the sender must time out
    # rather than silently proceeding to GRIP CLOSE.
    sender, conn = _sender(["DONE SELECT", "DONE Z"])
    with pytest.raises(TimeoutError, match="no reply"):
        sender.send(COMMAND)
    assert conn.written == ["SELECT SILICONE\n", "Z 84.2\n", "FORCE 1.75\n"]
