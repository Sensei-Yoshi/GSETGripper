# Gripper Firmware Design — 4-Motor `main.ino`

**Status:** Approved for implementation planning
**Scope:** `arduino/main/main.ino` only. Python-side command generation (vision classification, height computation, width computation) is out of scope — some of those values aren't ready yet. This spec only defines the firmware and the serial protocol it exposes.

## Context

This is the firmware for an automated pick system: a camera-based vision model classifies an object and picks a gripper type (gecko-adhesive or silicone), a gantry lowers to the object's height, the gripper closes around it, and the gantry lifts the object. Python scripts (already partially built — see `camera/depth_closest_read.py` for the existing Z-height pipeline) compute the values needed for each step and send them to an Arduino Mega over serial. The Mega's only job is to reliably execute motor moves on command; all decision-making (what to pick, which gripper, when) lives in Python.

The existing `main.ino` already implements this pattern for a single Z-axis stepper (`Z <mm>` command via `AccelStepper`). This spec extends that pattern from 1 motor to 4, without changing its fundamental shape.

## Goals

- Drive 4 stepper motors on an Arduino Mega: 2 lead-screw motors (mirrored Z axis), 1 gripper-type selector motor, 1 gripper open/close motor.
- Firmware is a "dumb executor": it has no onboard sequencing/state machine. Python issues one command, waits for confirmation the move finished, then issues the next. This keeps sequence logic (the order of select → lower → close → raise) entirely in Python, where it's easy to change without reflashing.
- Extend the existing serial line-protocol convention rather than replacing it.

## Non-goals

- Homing / limit switches. All 4 axes remain open-loop (position = wherever the motor was at power-on/reset), matching the current Z behavior. Worth revisiting once hardware is finalized, but not now.
- Exact calibration values (steps/mm, named-position step counts, grip width→steps formula). These are marked as `TODO: calibrate` constants, same convention as the existing `STEPS_PER_MM`.
- Concurrent multi-axis motion. Only one axis moves at a time under this protocol; the design doesn't preclude concurrent motion later, but doesn't build for it now (YAGNI).
- Any Python-side changes beyond what's needed to describe the protocol they'll speak.

## Architecture

Four `AccelStepper` instances, all STEP/DIR driver mode (matching the current hardware/driver assumption), each with its own speed/acceleration/calibration constants:

| Axis (variable name) | Purpose | STEP pin | DIR pin |
|---|---|---|---|
| `stepperZA` | Lead screw, side A (existing motor) | 2 | 3 |
| `stepperZB` | Lead screw, side B (new; always mirrors ZA) | 4 | 5 |
| `stepperSelect` | Cycles gripper head between gecko and silicone | 6 | 7 |
| `stepperGrip` | Opens/closes the currently active gripper | 8 | 9 |

`stepperZA` and `stepperZB` are two independent `AccelStepper` objects that always receive identical `moveTo()` targets and identical speed/acceleration — they are not merged into a single object or `MultiStepper` group. This keeps the existing single-Z code path recognizable and avoids taking on `MultiStepper`'s complexity for a case that's just "two motors, same target."

`loop()` calls `.run()` on all 4 steppers every iteration unconditionally (cheap no-op when idle), even though only one axis moves at a time under the current protocol — this keeps the 4 axes structurally uniform and leaves room for concurrent motion later without restructuring `loop()`.

## Serial Protocol

Line-terminated (`\n`) ASCII commands at 9600 baud (unchanged), same parsing convention as the current `processCommand()`.

| Command | Effect | Response on completion |
|---|---|---|
| `Z <mm>` | Move `stepperZA` and `stepperZB` to the same target height (mirrored) | `DONE Z` |
| `SELECT GEKKO` | Move `stepperSelect` to the calibrated gecko position | `DONE SELECT` |
| `SELECT SILICONE` | Move `stepperSelect` to the calibrated silicone position | `DONE SELECT` |
| `GRIP OPEN` | Move `stepperGrip` to the calibrated fully-open position | `DONE GRIP` |
| `GRIP CLOSE <mm>` | Close `stepperGrip` around an object of the given width | `DONE GRIP` |

Notes:
- `DONE <AXIS>` is printed once `distanceToGo() == 0` for the relevant stepper(s) (both ZA and ZB for `Z`).
- `GRIP CLOSE <mm>` converts width → target steps through a single, clearly isolated function (e.g. `gripStepsForWidth(float widthMM)`), since the exact relationship (jaw travel vs. object width) depends on gripper geometry that isn't finalized. The rest of the firmware only calls this function, so recalibrating later is a one-place edit.
- Named positions (`SELECT GEKKO/SILICONE`, `GRIP OPEN`) are stored as `TODO: calibrate` step-count constants, same convention as the current `STEPS_PER_MM`.

## Error Handling & Safety

- Unrecognized command verb, missing argument, or non-numeric value where a number is expected → firmware prints `ERR <reason>` (e.g. `ERR unknown command`, `ERR bad value`), issues no move, sends no `DONE`, and continues processing subsequent lines normally. A malformed line never wedges the firmware.
- Every axis has soft min/max step clamps (constants, independent of physical limit switches). A target outside the clamped range is clamped rather than rejected, and the firmware prints `WARN <axis> clamped to <value>` — the move still completes and still sends `DONE`, so Python isn't blocked by a clamp, but the clamp is visible for debugging.
- Python is expected to implement its own timeout while waiting for `DONE` (e.g. "no DONE within N seconds = treat as failure"); the firmware itself has no concept of timeouts — it only ever responds to what it's told.

## Testing Plan

All testable via the Arduino Serial Monitor, no Python or vision pipeline required:

1. Bench-test each axis independently: `Z 50`, `SELECT GEKKO`, `SELECT SILICONE`, `GRIP OPEN`, `GRIP CLOSE 40` — confirm correct motor movement and matching `DONE <AXIS>` output.
2. Confirm `stepperZA`/`stepperZB` move in mechanical lockstep under load (no observable racking between the two sides).
3. Feed malformed input (`Z abc`, `GRIP CLOSE` with no value, `FOO`, empty line) and confirm `ERR <reason>` responses with no `DONE` and no firmware hang.
4. Command a target beyond each axis's soft limits and confirm `WARN <axis> clamped to <value>` plus a `DONE` at the clamped position, not the requested one.

## Open Questions / Follow-ups (explicitly deferred)

- Real calibration values for all 4 axes (steps/mm, named-position step counts, grip width→steps formula) — depends on finalized mechanical hardware.
- Whether any axis eventually gets a physical limit switch / homing routine.
- The Python-side orchestration script that will actually sequence `SELECT` → `Z` → `GRIP CLOSE` → `Z` calls — out of scope for this spec.
