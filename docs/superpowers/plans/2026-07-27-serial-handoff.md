# Serial Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a predicted grasp (height, gripper, force, width) into validated serial commands and send them to `arduino/main/main.ino`.

**Architecture:** A pure `modules/handoff.py` owns a `GraspCommand` Pydantic model that gathers all four values, refuses to build unless every one is present and the grasp is feasible, and renders the firmware's command lines. A `SerialGraspSender` in `modules/hardware.py` does the I/O — writing one line, waiting for its `DONE` ack, then the next. The purity split lets unit tests pin the exact bytes with no Arduino attached.

**Tech Stack:** Python 3.11, Pydantic v2, pytest, pyserial (`types-pyserial` already in `pyproject.toml:33`).

**Spec:** `docs/superpowers/specs/2026-07-27-serial-handoff-design.md`

## Global Constraints

- All work happens inside `Force-Prediction/`. Paths below are relative to that directory.
- Ruff: line length 100, target py311, rules `E,F,I,UP,B` (`pyproject.toml:38-44`).
- Every module starts with `from __future__ import annotations`, matching every existing module.
- No tunable is hard-coded in Python — force bounds come from `Config`, never literals (`config.yaml:7`).
- Firmware geometry constants (`GRIP_MAX_OPENING_MM`, the `GEKKO` spelling) are hand-mirrored in one commented block. They are hardware facts, not tunables — they do **not** go in `config.yaml`.
- Millimetre values serialize at **1 decimal**; force at **2 decimals**. Fixed-point only, never scientific notation.
- `CLAUDE.md:101` defines `$VENV` as `/Users/premshah/...` — that is a different machine. Use this machine's Python environment. Verification commands below use plain `python`.
- `mypy` 2.3.0 may exit with its own internal error. Distinguish a tool crash from project diagnostics (`CLAUDE.md:112-113`).

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `modules/handoff.py` | create | `GraspCommand` model, validation, `serialize()`. Pure — no I/O, no serial import. |
| `tests/test_handoff.py` | create | Format, ordering, spelling, and every rejection rule. |
| `modules/hardware.py` | modify | Add `SerialGraspSender`; annotate `SerialGripper`'s docstring. |
| `tests/test_hardware_sender.py` | create | Ack/error/warn/timeout handling against a fake connection. |
| `tests/conftest.py` | modify | Autouse guard blocking real serial ports. |

---

## Task 1: `GraspCommand` model and `serialize()`

**Files:**
- Create: `Force-Prediction/modules/handoff.py`
- Test: `Force-Prediction/tests/test_handoff.py`

**Interfaces:**
- Consumes: `Gripper` from `modules/contracts.py:23`.
- Produces: `GraspCommand(object_id: str, object_height_mm: float, object_width_mm: float, gripper: Gripper, force_n: float)` with `serialize() -> list[str]`; module constants `GRIP_MAX_OPENING_MM: float` and `GECKO_SELECT_TOKEN: str`. Task 2 adds a classmethod to this same class; Task 3's sender calls `serialize()`.

- [ ] **Step 1: Write the failing tests**

Create `Force-Prediction/tests/test_handoff.py`:

```python
from __future__ import annotations

import pytest
from pydantic import ValidationError

from modules.contracts import Gripper
from modules.handoff import GraspCommand


def _command(**overrides) -> GraspCommand:
    values = {
        "object_id": "banana",
        "object_height_mm": 84.2,
        "object_width_mm": 61.0,
        "gripper": Gripper.SILICONE,
        "force_n": 1.75,
    }
    values.update(overrides)
    return GraspCommand(**values)


def test_serialize_emits_exact_firmware_lines():
    assert _command().serialize() == [
        "Z 84.2",
        "SELECT SILICONE",
        "FORCE 1.75",
        "GRIP CLOSE 61.0",
    ]


def test_gecko_serializes_with_the_firmware_spelling():
    # main.ino:223 compares against "GEKKO", not "GECKO". Sending the Python
    # enum value verbatim yields "ERR unknown select position".
    lines = _command(gripper=Gripper.GECKO).serialize()
    assert "SELECT GEKKO" in lines
    assert "SELECT GECKO" not in lines


def test_descends_before_closing_and_sets_force_first():
    lines = _command().serialize()
    z = next(i for i, line in enumerate(lines) if line.startswith("Z "))
    force = next(i for i, line in enumerate(lines) if line.startswith("FORCE "))
    grip = next(i for i, line in enumerate(lines) if line.startswith("GRIP CLOSE "))
    assert z < grip
    assert force < grip


def test_small_force_never_renders_in_scientific_notation():
    # parseFloatStrict (main.ino:340-371) accepts only sign, digits, one dot.
    # "1e-05" comes back as "ERR bad value".
    #
    # Check the ARGUMENT, not the whole line: FORCE, SILICONE, and CLOSE all
    # contain "e", so scanning the line for "e" can never pass.
    lines = _command(force_n=0.00001).serialize()
    force_line = next(line for line in lines if line.startswith("FORCE "))
    assert force_line == "FORCE 0.00"
    assert "e" not in force_line.removeprefix("FORCE ")


def test_rejects_nonpositive_dimensions():
    with pytest.raises(ValidationError):
        _command(object_height_mm=0.0)
    with pytest.raises(ValidationError):
        _command(object_width_mm=-1.0)


def test_rejects_negative_force():
    with pytest.raises(ValidationError):
        _command(force_n=-0.1)


def test_rejects_object_wider_than_the_jaws():
    with pytest.raises(ValueError, match="exceeds jaw opening"):
        _command(object_width_mm=150.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd Force-Prediction && python -m pytest tests/test_handoff.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modules.handoff'`

- [ ] **Step 3: Write the implementation**

Create `Force-Prediction/modules/handoff.py`:

```python
"""Turn one predicted grasp into the serial lines main.ino executes.

Pure by design: this module never opens a serial port. `SerialGraspSender` in
:mod:`modules.hardware` does the I/O, so the exact bytes sent to the rig can be
pinned by unit tests with no Arduino attached.

See docs/superpowers/specs/2026-07-27-serial-handoff-design.md.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from .contracts import Gripper

# --------------------------------------------------------------------------- #
# Mirrors of arduino/main/main.ino constants. Keep in sync by hand.
# These are hardware facts, not tunables -- they do not belong in config.yaml.
# --------------------------------------------------------------------------- #
GRIP_MAX_OPENING_MM = 101.6    # main.ino:131  GRIP_MAX_OPENING_MM
GECKO_SELECT_TOKEN = "GEKKO"   # main.ino:223  firmware's spelling of "gecko"

_SELECT_TOKENS = {
    Gripper.GECKO: GECKO_SELECT_TOKEN,
    Gripper.SILICONE: "SILICONE",
}


class GraspCommand(BaseModel):
    """One executable grasp: every value the firmware needs, already validated.

    `gripper` is typed as `Gripper` rather than `GripperChoice` on purpose --
    `GripperChoice` admits "none", and an unrepresentable state beats a
    rejected one.
    """

    object_id: str = Field(min_length=1)
    object_height_mm: float = Field(gt=0)
    object_width_mm: float = Field(gt=0)
    gripper: Gripper
    force_n: float = Field(ge=0)

    @model_validator(mode="after")
    def _width_fits_the_jaws(self) -> GraspCommand:
        # The firmware's response to an over-wide object is silent: it floors
        # negative travel to zero (main.ino:317-320) and leaves the jaws fully
        # open with no WARN. Raising beats a rig that looks like it is working.
        if self.object_width_mm > GRIP_MAX_OPENING_MM:
            raise ValueError(
                f"width {self.object_width_mm} mm exceeds jaw opening "
                f"{GRIP_MAX_OPENING_MM} mm"
            )
        return self

    def serialize(self) -> list[str]:
        """Render the firmware command lines, in execution order.

        Z precedes GRIP CLOSE so the gripper descends before the jaws close --
        reversed, the jaws close in mid-air and the gripper descends already
        shut. FORCE precedes GRIP CLOSE so the target is set before any jaw
        motion begins.

        Fixed-point formatting is mandatory: parseFloatStrict (main.ino:340-371)
        accepts only an optional sign, digits, and at most one dot. Python's
        default float formatting emits "1e-05" for small values, which the
        firmware answers with "ERR bad value".
        """
        return [
            f"Z {self.object_height_mm:.1f}",
            f"SELECT {_SELECT_TOKENS[self.gripper]}",
            f"FORCE {self.force_n:.2f}",
            f"GRIP CLOSE {self.object_width_mm:.1f}",
        ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd Force-Prediction && python -m pytest tests/test_handoff.py -v`
Expected: PASS — 7 passed

- [ ] **Step 5: Lint**

Run: `cd Force-Prediction && python -m ruff check modules/handoff.py tests/test_handoff.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add Force-Prediction/modules/handoff.py Force-Prediction/tests/test_handoff.py
git commit -m "feat: add GraspCommand and firmware line serialization"
```

---

## Task 2: `from_prediction` adapter

**Files:**
- Modify: `Force-Prediction/modules/handoff.py` (add a classmethod to `GraspCommand`)
- Test: `Force-Prediction/tests/test_handoff.py` (append)

**Interfaces:**
- Consumes: `GraspCommand` from Task 1; `SelectionResult` from `modules/contracts.py:193`; `DatasetObject` and `ContactFractionArtifact` from `modules/datasets/models.py:82,61`; `Config` from `modules/config.py:182`.
- Produces: `GraspCommand.from_prediction(obj: DatasetObject, selection: SelectionResult, cfg: Config) -> GraspCommand`, raising `ValueError` when the grasp cannot be executed.

- [ ] **Step 1: Extend the import block at the top of the test file**

Do **not** append these imports at the bottom — ruff's `E402` (module-level import not at top of file) and `I` (import sorting) are both enabled (`pyproject.toml:43`). Replace the existing import block at the top of `Force-Prediction/tests/test_handoff.py` with:

```python
from __future__ import annotations

import pytest
from pydantic import ValidationError

from modules.config import load_config
from modules.contracts import Gripper, SelectionResult
from modules.datasets.models import (
    ContactFractionArtifact,
    DatasetObject,
    ImageArtifact,
)
from modules.handoff import GraspCommand

CFG = load_config()
```

- [ ] **Step 2: Write the failing tests**

Append to the bottom of `Force-Prediction/tests/test_handoff.py`:

```python
def _contact(**overrides) -> ContactFractionArtifact:
    values = {
        "summary_path": "data/expforce/objects/banana/contact_fraction/summary.json",
        "schema_version": 2,
        "object_height_mm": 84.2,
        "object_width_mm": 61.0,
        "geometric_contact_fraction": 0.42,
        "combined_contact_fraction": 0.42,
        "grasp_feasible": True,
        "antipodal_grasp": True,
        "contact_floor_applied": False,
    }
    values.update(overrides)
    return ContactFractionArtifact(**values)


def _object(contact: ContactFractionArtifact | None = None) -> DatasetObject:
    return DatasetObject(
        dataset_id="expforce",
        object_id="banana",
        name="Banana",
        image=ImageArtifact(path="data/expforce/objects/banana/image.png"),
        contact_fraction=contact if contact is not None else _contact(),
    )


def _selection(**overrides) -> SelectionResult:
    values = {"desired_gripper": "silicone", "predicted_normal_force_n": 1.75}
    values.update(overrides)
    return SelectionResult(**values)


def test_from_prediction_assembles_all_four_values():
    cmd = GraspCommand.from_prediction(_object(), _selection(), CFG)
    assert cmd.object_id == "banana"
    assert cmd.object_height_mm == 84.2
    assert cmd.object_width_mm == 61.0
    assert cmd.gripper is Gripper.SILICONE
    assert cmd.force_n == 1.75
    assert cmd.serialize()[0] == "Z 84.2"


def test_from_prediction_rejects_infeasible_selection():
    selection = _selection(desired_gripper="none", predicted_normal_force_n=None)
    with pytest.raises(ValueError, match="no feasible gripper for banana"):
        GraspCommand.from_prediction(_object(), selection, CFG)


def test_from_prediction_rejects_missing_force():
    selection = _selection(predicted_normal_force_n=None)
    with pytest.raises(ValueError, match="selection has no force for banana"):
        GraspCommand.from_prediction(_object(), selection, CFG)


def test_from_prediction_rejects_missing_contact_data():
    obj = DatasetObject(
        dataset_id="expforce",
        object_id="banana",
        name="Banana",
        image=ImageArtifact(path="data/expforce/objects/banana/image.png"),
    )
    with pytest.raises(ValueError, match="run prepare_dataset --stages surface_area"):
        GraspCommand.from_prediction(obj, _selection(), CFG)


def test_from_prediction_rejects_infeasible_contact_geometry():
    obj = _object(_contact(grasp_feasible=False))
    with pytest.raises(ValueError, match="no antipodal grasp for banana"):
        GraspCommand.from_prediction(obj, _selection(), CFG)


def test_from_prediction_rejects_force_over_the_configured_limit():
    selection = _selection(predicted_normal_force_n=CFG.force.limit_n + 0.5)
    with pytest.raises(ValueError, match="exceeds limit"):
        GraspCommand.from_prediction(_object(), selection, CFG)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cd Force-Prediction && python -m pytest tests/test_handoff.py -v -k from_prediction`
Expected: FAIL — `AttributeError: type object 'GraspCommand' has no attribute 'from_prediction'`

- [ ] **Step 4: Add the imports to `handoff.py`**

In `Force-Prediction/modules/handoff.py`, replace the single contracts import with:

```python
from .config import Config
from .contracts import Gripper, SelectionResult
from .datasets.models import DatasetObject
```

- [ ] **Step 5: Add the classmethod**

Append inside `class GraspCommand`, after `_width_fits_the_jaws` and before `serialize`:

```python
    @classmethod
    def from_prediction(
        cls,
        obj: DatasetObject,
        selection: SelectionResult,
        cfg: Config,
    ) -> GraspCommand:
        """Assemble a command from a pipeline selection and an object's contact data.

        Gripper and force come from `Pipeline.predict()` (pipeline.py:33); height
        and width from the contact model's summary.json, surfaced as
        `DatasetObject.contact_fraction` (catalog.py:366). Those two halves have
        different lifecycles -- one live, one read from disk -- which is why they
        are validated together here.

        Raises ValueError naming the fix when the grasp cannot be executed.
        All-or-nothing: nothing partial reaches a rig that has no homing routine
        and no limit switches.
        """
        object_id = obj.object_id
        if selection.desired_gripper == "none":
            raise ValueError(f"no feasible gripper for {object_id}")
        force_n = selection.predicted_normal_force_n
        if force_n is None:
            raise ValueError(f"selection has no force for {object_id}")
        contact = obj.contact_fraction
        if contact is None:
            raise ValueError(
                f"no contact data for {object_id}: "
                "run prepare_dataset --stages surface_area"
            )
        if not contact.grasp_feasible:
            raise ValueError(f"contact model found no antipodal grasp for {object_id}")
        if force_n > cfg.force.limit_n:
            raise ValueError(f"force {force_n} N exceeds limit {cfg.force.limit_n} N")
        return cls(
            object_id=object_id,
            object_height_mm=contact.object_height_mm,
            object_width_mm=contact.object_width_mm,
            gripper=Gripper(selection.desired_gripper),
            force_n=force_n,
        )
```

- [ ] **Step 6: Run the full handoff test file**

Run: `cd Force-Prediction && python -m pytest tests/test_handoff.py -v`
Expected: PASS — 13 passed

- [ ] **Step 7: Lint and type-check**

Run: `cd Force-Prediction && python -m ruff check modules/handoff.py tests/test_handoff.py && python -m mypy modules/handoff.py`
Expected: ruff `All checks passed!`; mypy clean (or its own internal crash — that is a tool failure, not a project diagnostic)

- [ ] **Step 8: Commit**

```bash
git add Force-Prediction/modules/handoff.py Force-Prediction/tests/test_handoff.py
git commit -m "feat: assemble GraspCommand from a pipeline selection"
```

---

## Task 3: Serial sender and test guard

**Files:**
- Modify: `Force-Prediction/modules/hardware.py` (add `SerialGraspSender`; annotate `SerialGripper` docstring at line 102)
- Modify: `Force-Prediction/tests/conftest.py` (add autouse serial guard)
- Test: `Force-Prediction/tests/test_hardware_sender.py`

**Interfaces:**
- Consumes: `GraspCommand.serialize()` from Task 1; `find_serial_port()` from `modules/hardware.py:79`.
- Produces: `SerialGraspSender(port: str | None = None, baud: int = 9600, timeout: float = 30.0)` with `send(cmd: GraspCommand) -> list[str]` returning firmware `WARN` lines, and `close() -> None`. `GraspCommand` is imported under a `TYPE_CHECKING` guard — annotation only, no runtime import.

- [ ] **Step 1: Add the serial guard to conftest**

Append to `Force-Prediction/tests/conftest.py`:

```python
@pytest.fixture(autouse=True)
def block_real_serial_ports(monkeypatch):  # noqa: ANN001, ANN201
    """Mirror of the Gemini guard above, for hardware.

    A unit test that opened a real port could drive an unhomed rig into the
    table -- there is no limit switch to stop it (main.ino:58-62).
    """
    import serial

    def blocked(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError(
            "unit test attempted to open a real serial port; install a test fake"
        )

    monkeypatch.setattr(serial, "Serial", blocked)
```

- [ ] **Step 2: Write the failing tests**

Create `Force-Prediction/tests/test_hardware_sender.py`:

```python
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
    sender, conn = _sender(["DONE Z", "DONE SELECT", "DONE FORCE", "DONE GRIP"])
    warnings = sender.send(COMMAND)
    assert conn.written == [
        "Z 84.2\n",
        "SELECT SILICONE\n",
        "FORCE 1.75\n",
        "GRIP CLOSE 61.0\n",
    ]
    assert warnings == []


def test_error_raises_and_does_not_hang():
    # sendErr returns without setting a pending flag (main.ino:216-217), so no
    # DONE ever follows. A reader that waits for one would block forever.
    sender, conn = _sender(["ERR bad value"])
    with pytest.raises(RuntimeError, match="ERR bad value"):
        sender.send(COMMAND)
    assert conn.written == ["Z 84.2\n"]


def test_warning_is_collected_and_not_mistaken_for_the_ack():
    sender, _ = _sender(
        [
            "WARN Z clamped to 252.7 mm above floor",
            "DONE Z",
            "DONE SELECT",
            "DONE FORCE",
            "DONE GRIP",
        ]
    )
    warnings = sender.send(COMMAND)
    assert warnings == ["WARN Z clamped to 252.7 mm above floor"]


def test_silent_board_times_out():
    sender, _ = _sender([])
    with pytest.raises(TimeoutError, match="no reply"):
        sender.send(COMMAND)


def test_unexpected_reply_raises_rather_than_looping():
    sender, _ = _sender(["HELLO"])
    with pytest.raises(RuntimeError, match="unexpected firmware reply"):
        sender.send(COMMAND)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cd Force-Prediction && python -m pytest tests/test_hardware_sender.py -v`
Expected: FAIL — `ImportError: cannot import name 'SerialGraspSender' from 'modules.hardware'`

- [ ] **Step 4: Annotate the existing `SerialGripper`**

In `Force-Prediction/modules/hardware.py`, replace the one-line docstring at line 102:

```python
class SerialGripper:
    """Real gripper + load cell over one Arduino link (firmware/gripper_force).

    NOTE: this targets firmware that has NOT been written. `arduino/main/main.ino`
    speaks a different protocol (Z / SELECT / GRIP -- see `SerialGraspSender`
    below) and implements none of the commands used here.

    Kept because it is the only written record of what main.ino must still grow
    before `collect_real` can run on real hardware:
        SET_FORCE <n>   set the stationary-finger target force
        CLOSE           close until contact
        OPEN            release
        READ            report the load-cell reading in newtons
        LIFT            perform the standardized lift; reply HELD or not
    """
```

- [ ] **Step 5: Add the sender**

First extend the existing `typing` import at `Force-Prediction/modules/hardware.py:23`:

```python
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .handoff import GraspCommand
```

A `TYPE_CHECKING` guard, not a plain import: the annotation is already a string thanks to `from __future__ import annotations`, so this gives full typing without pulling `handoff` — and its `datasets.models` → `perception` chain — into `hardware`'s runtime imports. There is no cycle (nothing in `handoff` imports `hardware`); this is about import weight, since `collect.py:31` imports `hardware` eagerly.

Then append after `SerialRoughness`:

```python
class SerialGraspSender:
    """Send one GraspCommand to arduino/main/main.ino, one line at a time.

    Writes a line, waits for its DONE ack, then writes the next -- rather than
    dumping all four. The firmware sets a pending flag and prints DONE <axis>
    only once distanceToGo() reaches 0 (main.ino:192-203), so waiting is what
    keeps the moves sequential instead of overlapping.

    PHYSICAL PRECONDITION: opening the port resets the board, and the firmware
    then ASSUMES the carriage is parked at the top of travel and the jaws are
    fully open. There is no homing routine and no limit switch (main.ino:58-62,
    126-129). Park the rig before constructing this, or the gripper is driven
    into the table with nothing in hardware to stop it.
    """

    def __init__(
        self,
        port: str | None = None,
        baud: int = 9600,
        timeout: float = 30.0,
    ) -> None:
        import serial

        self.port = find_serial_port(port)
        self.conn = serial.Serial(self.port, baud, timeout=timeout)
        time.sleep(2.0)  # allow the board to reset

    def send(self, cmd: GraspCommand) -> list[str]:
        """Execute the grasp. Returns any WARN lines the firmware emitted.

        A WARN means the firmware clamped something -- the rig did not do
        exactly what was asked, so callers should surface these.
        """
        warnings: list[str] = []
        for line in cmd.serialize():
            self.conn.write((line + "\n").encode("ascii"))
            self.conn.flush()
            warnings.extend(self._await_ack(line))
        return warnings

    def _await_ack(self, line: str) -> list[str]:
        warnings: list[str] = []
        while True:
            reply = self.conn.readline().decode("ascii", errors="ignore").strip()
            if not reply:
                raise TimeoutError(f"no reply from firmware for {line!r}")
            if reply.startswith("ERR"):
                # processCommand returns after sendErr without setting a pending
                # flag (main.ino:216-217), so no DONE will ever follow. Return
                # now; waiting for one blocks forever.
                raise RuntimeError(f"firmware rejected {line!r}: {reply}")
            if reply.startswith("WARN"):
                warnings.append(reply)
                continue
            if reply.startswith("DONE"):
                return warnings
            raise RuntimeError(f"unexpected firmware reply to {line!r}: {reply}")

    def close(self) -> None:
        self.conn.close()
```

- [ ] **Step 6: Run the sender tests**

Run: `cd Force-Prediction && python -m pytest tests/test_hardware_sender.py -v`
Expected: PASS — 5 passed

- [ ] **Step 7: Run the whole suite**

Run: `cd Force-Prediction && python -m pytest`
Expected: All tests pass. The new autouse serial guard now applies to every test — if `tests/test_collect.py` fails, it is reaching a real port and its fakes need checking.

- [ ] **Step 8: Lint and type-check**

Run: `cd Force-Prediction && python -m ruff check . && python -m mypy modules`
Expected: ruff `All checks passed!`; mypy clean (or its own internal crash — a tool failure, not a project diagnostic)

- [ ] **Step 9: Commit**

```bash
git add Force-Prediction/modules/hardware.py Force-Prediction/tests/test_hardware_sender.py Force-Prediction/tests/conftest.py
git commit -m "feat: add SerialGraspSender and block real ports in tests"
```

---

## Done criteria

- `python -m pytest` passes from `Force-Prediction/`.
- `python -m ruff check .` is clean.
- `GraspCommand.from_prediction(obj, Pipeline(cfg, "e4").fit(train).predict(q), cfg).serialize()` returns four lines, or raises a `ValueError` naming the missing piece.
- No test opens a serial port.

## Known blockers beyond this plan

These are expected, not defects — both are recorded in the spec.

1. **No object has height or width yet.** `from_prediction` raises for all 129 until `scripts/calibrate_scale.py` then `scripts/prepare_dataset.py --stages surface_area` have run. Nothing in this plan produces that data.
2. **`FORCE` does not exist in the firmware.** Until `main.ino` implements it, a real send fails at the third line with `RuntimeError: firmware rejected 'FORCE 1.75': ERR unknown command` (`main.ino:254`). Tasks 1–3 are still fully testable, because every test uses a fake connection.

## Note from the force-grasp firmware work (2026-07-27)

The merged firmware prints boot lines (`INFO`/`READY`, and `ERR cell <n> timed
out` on a wiring fault) before accepting commands. `SerialGraspSender` must
call `reset_input_buffer()` after its 2.0 s post-open sleep, or the first
`Z` ack read will consume boot output — and a boot `ERR` line would be
mistaken for a command failure.
