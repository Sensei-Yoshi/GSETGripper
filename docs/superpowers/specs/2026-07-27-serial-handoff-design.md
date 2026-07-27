# Serial Handoff Design — Force-Prediction to `main.ino`

**Status:** Approved for implementation planning
**Scope:** `Force-Prediction/modules/handoff.py` (new), `Force-Prediction/modules/hardware.py` (extended), plus `tests/test_handoff.py` (new) and an autouse guard in `tests/conftest.py`. Firmware changes are out of scope — this spec defines the protocol `main.ino` will be written to match, and records the firmware work it implies as follow-ups.

## Context

`Force-Prediction` predicts which of two compliant grippers to use and the minimum normal force to lift an object. `arduino/main/main.ino` executes motor moves on command. Nothing currently connects them: the prediction stays in Python, and the firmware has never received a value from it.

This spec defines that connection — a Python-side module that gathers the values describing one grasp, validates them as a set, and turns them into the serial lines the firmware executes.

The companion firmware spec is `2026-07-21-gripper-firmware-design.md`, which established the line protocol this extends.

## Goals

- Assemble one grasp's values from the two places they live, in a single well-defined place.
- Make a partial or infeasible grasp impossible to transmit.
- Keep command generation pure and testable with no Arduino attached.
- Extend the firmware's existing line protocol rather than replace it.

## Non-goals

- **Firmware changes.** `FORCE` does not exist in `main.ino` yet. This spec defines its wire format; implementing it is separate work.
- **Load-cell closed-loop control.** The load cell is physically wired but uncoded. `FORCE` transmits a target; what the firmware does with it is later work.
- **Producing height and width data.** This design consumes the contact model's output. It does not run, fix, or calibrate it.
- **Grasp sequencing beyond one object.** No pick-and-place loop, no queue, no retry. One grasp, one transmission.
- **Replacing `SerialGripper`.** See "Existing code" below.

## Architecture

Two files, split along a purity boundary.

| File | Status | Responsibility |
|---|---|---|
| `modules/handoff.py` | new | Gather 4 values, validate, render command lines. No I/O. |
| `modules/hardware.py` | extended | Open the port, write lines, wait for acks. |

The split exists so the exact bytes sent to the rig can be pinned by unit tests with no hardware present. Hardware is needed only to test transmission, never to test the format.

### `GraspCommand` (in `handoff.py`)

A Pydantic model, consistent with `contracts.py` owning all shared shapes and forbidding ad-hoc dicts.

| Field | Type | Constraint |
|---|---|---|
| `object_id` | `str` | non-empty |
| `object_height_mm` | `float` | `> 0` |
| `object_width_mm` | `float` | `> 0`, `<= GRIP_MAX_OPENING_MM` |
| `gripper` | `Gripper` | `gecko` or `silicone` |
| `force_n` | `float` | `>= 0`; upper bound checked against config, see Validation |

`gripper` is typed as the `Gripper` enum, **not** `GripperChoice`. `GripperChoice` admits `"none"`; using `Gripper` makes "no feasible gripper" unrepresentable rather than merely rejected.

Two construction paths, both running the same validators:

- `GraspCommand(...)` — direct, four explicit values. Used by tests.
- `GraspCommand.from_prediction(obj, selection, cfg)` — the adapter. Used by callers.

This mirrors `query_input_from_object()` (`modules/pipeline.py:40`), which adapts stored records into a typed input for the next stage. `from_prediction` is the same move at the pipeline's output end.

### Firmware mirror constants

`handoff.py` declares, in one block, the firmware limits it must respect:

```python
# Mirrors of arduino/main/main.ino constants. Keep in sync by hand.
GRIP_MAX_OPENING_MM = 101.6    # main.ino:131  GRIP_MAX_OPENING_MM
GECKO_SELECT_TOKEN = "GEKKO"   # main.ino:223  firmware spelling
```

Hand-synced duplication is a known cost, accepted because the alternatives are worse: parsing the `.ino` at runtime, or moving hardware geometry into `config.yaml` where it would be misread as a tunable. Confining it to one commented block keeps the sync surface visible. See Follow-ups.

## Serial protocol

Extends the existing line protocol: `\n`-terminated ASCII, 9600 baud.

| Command | Argument | Status |
|---|---|---|
| `Z <mm>` | object top height above the firmware floor | exists (`main.ino:211`) |
| `SELECT <GEKKO\|SILICONE>` | turret position | exists (`main.ino:220`) |
| `FORCE <newtons>` | target normal force | **new — firmware does not implement this yet** |
| `GRIP CLOSE <mm>` | jaw gap | exists (`main.ino:230`) |

`FORCE` is a separate top-level command rather than an extra argument on `GRIP CLOSE`, so force stays decoupled from jaw position. The load-cell loop can later change how force is achieved without altering the wire format, and the firmware parses it in the same `if/else` chain as the others.

### Emitted sequence

```
Z 84.2
SELECT SILICONE
FORCE 1.75
GRIP CLOSE 61.0
```

Ordering is load-bearing:

- `Z` before `GRIP CLOSE` — descend to the object, then close. Reversed, the jaws close in mid-air and the gripper descends already shut.
- `FORCE` before `GRIP CLOSE` — the target is set before any jaw motion, which is what a closed-loop implementation will need.

### Number formatting

Millimetre values render at **1 decimal**; force renders at **2 decimals**. Force resolution matches `collection.fine_step_n` (0.01 N, `config.yaml:37`); the firmware itself prints millimetres at 1 decimal (`main.ino:272`).

Fixed-point is mandatory. `parseFloatStrict` (`main.ino:340-371`) accepts only an optional sign, digits, and at most one dot — it rejects exponents. Python's default float formatting emits `1e-05` for small values, which the firmware answers with `ERR bad value`.

## Data flow

```
                    contact model (needs geometry.px_per_mm)
                              │
                              ▼
        data/<ds>/objects/<id>/contact_fraction/summary.json
                    results.object_height_mm ─────┐
                    results.object_width_mm  ───┐ │
                              │                 │ │
        DatasetObject.contact_fraction          │ │
          (catalog.py:366 _attach_contact_summary) │
                                                │ │
        Pipeline(cfg, exp).fit(train).predict(q)│ │
                    │  pipeline.py:33           │ │
                    ▼                           │ │
              SelectionResult                   │ │
                desired_gripper ────────────────┼─┤
                predicted_normal_force_n ───────┼─┤
                                                ▼ ▼
              GraspCommand.from_prediction(obj, selection, cfg)
                              │ serialize()
                              ▼
                    ["Z 84.2", "SELECT SILICONE",
                     "FORCE 1.75", "GRIP CLOSE 61.0"]
                              │ SerialGraspSender.send()
                              ▼
                          main.ino
```

### Sources

| Value | Source | Produced by |
|---|---|---|
| height | `obj.contact_fraction.object_height_mm` | contact model, offline, written to disk |
| width | `obj.contact_fraction.object_width_mm` | contact model, offline, written to disk |
| gripper | `selection.desired_gripper` | `prediction.py:135` `select()`, live |
| force | `selection.predicted_normal_force_n` | `prediction.py:135` `select()`, live |

The halves have different lifecycles: gripper and force come from a live call, height and width from a file a separate stage must already have written. `from_prediction` validating both at once is the guard against that asymmetry.

`SelectionResult` is the input type, not `PipelineRunResult` — the narrower contract. Experiment provenance is not carried into the command.

Height needs no coordinate transform: `object_height_mm` and the firmware's `Z <mm>` argument both mean the object's top above the firmware floor.

## Validation

`from_prediction` raises `ValueError` naming the fix. All-or-nothing: nothing partial can reach a rig that has no homing routine and no limit switches.

| Condition | Message |
|---|---|
| `desired_gripper == "none"` | `no feasible gripper for <id>` |
| `predicted_normal_force_n is None` | `selection has no force for <id>` |
| `contact_fraction is None` | `no contact data for <id>: run prepare_dataset --stages surface_area` |
| `contact_fraction.grasp_feasible is False` | `contact model found no antipodal grasp for <id>` |
| `force_n > cfg.force.limit_n` | `force <n> N exceeds limit <limit> N` |
| `object_width_mm > GRIP_MAX_OPENING_MM` | `width <w> mm exceeds jaw opening 101.6 mm` |

The static lower bounds (`force_n >= 0`, dimensions `> 0`) are Pydantic `Field` constraints, so they hold for direct construction too. The `cfg.force.limit_n` upper bound is checked in `from_prediction`, which is where config is available.

The width check exists because the firmware's response to an over-wide object is silent: `gripStepsForWidth` floors negative travel to zero (`main.ino:317-320`), leaving the jaws fully open with no `WARN`. A Python exception beats a rig that appears to be working.

This mirrors `ExperienceRecord._feasibility_consistency` (`contracts.py:68`), which already refuses to build inconsistent records.

## Sender behavior

`SerialGraspSender` in `hardware.py`. Writes one line, waits for its ack, then writes the next — rather than dumping all four. The firmware sets a pending flag and prints `DONE <axis>` only when `distanceToGo() == 0` (`main.ino:192-203`), so waiting is what keeps moves sequential instead of overlapping.

Three reply shapes must be distinguished:

| Reply | Handling |
|---|---|
| `DONE <axis>` | The ack. Proceed to the next line. |
| `ERR <reason>` | Raise immediately. **No `DONE` will ever follow** — `processCommand` returns after `sendErr` without setting a pending flag (`main.ino:216-217`, `241`, `247`). A reader that blocks on "read until DONE" hangs forever. |
| `WARN ... clamped to ...` | Neither ack nor error; printed before the eventual `DONE` (`main.ino:271-273`, `332-335`). Collect and return — a clamp means the rig did something other than what was asked. |

A read timeout raises, so a dead board fails loudly rather than blocking.

Reuses `find_serial_port()` (`hardware.py:79`) for port autodetection.

### Reset precondition

Opening the serial connection resets the Arduino. The firmware then **assumes** the carriage is parked at the top of travel and the jaws fully open — there is no homing and no limit switch (`main.ino:58-62`, `126-129`). The sender sleeps 2.0 s after opening, as `SerialGripper` does (`hardware.py:110`).

This precondition must be stated in the sender's docstring. Violating it drives the gripper into the table with nothing in hardware to stop it.

## Existing code

`SerialGripper` (`hardware.py:101-137`) speaks a different protocol — `SET_FORCE` / `OPEN` / `CLOSE` / `READ` / `LIFT` — against a `firmware/gripper_force` sketch that does not exist in this repo.

**It stays.** It is not dead code: `collect_real()` (`collect.py:279-288`) uses it as both gripper and load cell for the real-hardware collection flow, and `tests/test_collect.py:91` monkeypatches it. More importantly, it is the only written record of the `READ` and `LIFT` commands that `main.ino` must grow before real data collection can run.

Its docstring gains a note stating that it targets firmware not yet written, and listing the commands `main.ino` still owes it.

## Testing

New `tests/test_handoff.py`, pure, no hardware:

1. Golden line format — exact list equality, not substring matching.
2. `Gripper.GECKO` serializes to `SELECT GEKKO`. Its own test, because a later "spelling fix" on one side silently breaks selection.
3. Ordering — `Z` precedes `GRIP CLOSE`; `FORCE` precedes `GRIP CLOSE`.
4. One test per validation rule above.
5. A very small force renders as `0.00`, never `1e-05`.

Sender tests use a fake connection replaying scripted replies, following the monkeypatch precedent in `test_collect.py:88-95`:

| Scenario | Assertion |
|---|---|
| `DONE` per line | All four written, in order |
| `ERR bad value` | Raises, and does not hang |
| `WARN` then `DONE` | Warning returned; `WARN` not mistaken for the ack |
| Silence | Times out and raises |

`conftest.py` gains an autouse fixture blocking `serial.Serial`, mirroring the existing Gemini network guard (`conftest.py:10-20`). A unit test that accidentally opened a real port could drive an unhomed rig into the table.

Verification (`CLAUDE.md:100-113`; mypy 2.3.0 may exit with its own internal error — distinguish a tool crash from project diagnostics):

```
$VENV -m pytest
$VENV -m ruff check .
$VENV -m mypy modules
```

## Prerequisites

**No object has height or width data today.** `find data -name '*summary*.json'` returns nothing across all 129 objects. `from_prediction` will raise for every one of them until the contact stage runs:

```
$VENV scripts/calibrate_scale.py          # establishes geometry.px_per_mm
$VENV scripts/prepare_dataset.py --dataset <id> --stages surface_area
```

This is consumed, not produced, by this design.

## Firmware follow-ups

Out of scope here; recorded because this design implies them.

1. ~~**`FORCE <n>` does not exist.**~~ **RESOLVED 2026-07-27.** The firmware now
   implements `FORCE <n>`: it parses with `parseFloatStrict`, rejects negative or
   malformed values via `sendErr`, treats `FORCE 0` as disarming, and emits a
   `DONE FORCE` ack. The Python states this as a protocol contract rather than a
   status claim, so it will not go stale again.

   **Still open:** `forceArmed` / `forceTargetN` are set by the `FORCE` handler but
   not yet read anywhere, so an armed force target does not currently gate
   `GRIP CLOSE`. Until it does, see follow-up 5.

5. **`GRIP CLOSE <width>` produces a zero-preload grasp by construction.**
   `gripStepsForWidth` sets the final jaw gap *equal* to its argument, and
   `object_width_mm` is `np.ptp(pts[:, 0])` — the object's *widest* cross-section.
   So the emitted command closes to exactly the widest point: contact without
   compression, zero normal force. A 1 mm overestimate from the contact model
   means the jaws never touch at all.

   This is a design gap in this spec, not an implementation defect — the code
   faithfully renders what was specified. Resolving it needs a decision:
   either `GRIP CLOSE` becomes force-terminated (reading `forceArmed`), or the
   handoff subtracts a compression allowance before emitting the width. The
   second needs a new tunable and belongs in `config.yaml`, unlike the firmware
   geometry constants.

6. **Nothing calls this code yet.** `handoff` is imported only by its tests. There
   is no `scripts/send_grasp.py`, no CLI, no Streamlit control. The plan scoped
   that out deliberately, but it means follow-up 5 and the floor-plane
   precondition stay theoretical until someone writes that caller — which is
   where both would surface first.

2. **Z flooring makes height a no-op for small objects.** `moveZForGrasp` computes `target = objectTop - GRASP_DEPTH_BELOW_TOP_MM` (107.95 mm) and floors it at `Z_MIN_GRIPPER_HEIGHT_MM` (10 mm) — `main.ino:290-296`. For any object shorter than **~118 mm**, the target floors out and the gripper goes to the same minimum height *regardless of the height sent*. No `WARN` fires, because the flooring happens before `moveZTo`'s clamp sees it.

   Most of the dataset is under 118 mm — a banana, a blueberry, a drinking straw. For those objects the transmitted height has no observable effect, so "height is being sent correctly" cannot be verified by watching the rig. Worth revisiting whether short objects should instead be grasped at a proportional depth.

3. **Load-cell integration.** Wired but uncoded. Deciding whether `FORCE` sets a target for a closed loop or is advisory will determine whether `GRIP CLOSE` remains position-controlled.

4. **Constant duplication.** `GRIP_MAX_OPENING_MM` and the `GEKKO` spelling exist in both `main.ino` and `handoff.py`. If this list grows, generating a shared header from one source becomes worth the effort.
