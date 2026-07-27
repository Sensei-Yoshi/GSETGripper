# Force-Controlled Grasp Design — Load Cells + `FORCE` in `main.ino`

**Status:** Approved for implementation planning
**Scope:** `arduino/main/main.ino` only. Integrates the HX711 load cells (currently a standalone sketch) and implements the `FORCE` command whose wire format `2026-07-27-serial-handoff-design.md` already defined. This is that spec's firmware follow-ups #1 and #3.

## Context

The Python pipeline emits `Z → SELECT → FORCE → GRIP CLOSE`, one line at a time, waiting for an ack after each (`2026-07-27-serial-handoff-design.md`). Today `main.ino` answers `FORCE` with `ERR unknown command` and closes the jaws purely by position. The load cells that could close the loop are wired but live in a separate load-cell-only sketch.

This design merges the load-cell reading path into `main.ino` and makes `GRIP CLOSE` force-seeking when a target is armed: creep closed until the selected gripper's load cell reads the target force, lift the object 50 mm, then ack.

## Goals

- Implement `FORCE <newtons>` exactly as the handoff spec transmits it.
- Close-until-force-reached using the load cell belonging to the selected gripper.
- Lift 50 mm automatically after a successful grasp, before the ack.
- Carry over the minimum load-cell utilities: `F?`, `TARE`, `CAL`, `CAL?`, EEPROM persistence.
- **Zero behavior change to every existing command.** `Z <mm>`, `SELECT <pos>`, `GRIP OPEN`, and `GRIP CLOSE <mm>` with no force target armed must behave exactly as they do today — same motion, same `WARN`/`ERR`/`DONE` lines, same ordering. The one-shot force target is the only thing that changes `GRIP CLOSE`, and only for the single command it arms.

## Non-goals

- Slip detection or force re-checking during the lift. If the object slips, it slips.
- Force maintenance (servoing) after the grasp. The lead screw holds position mechanically.
- `STREAM`, `MARK`, `SETTLE` — they stay in the standalone load-cell sketch.
- Multi-grasp sequencing. After the lift, Z sits 50 mm above what the last `Z` command set; a future loop must re-send `Z` per object.
- Python-side changes. The sender already sends the four lines and waits for `DONE` per line; the "Automatic, one ack" lift keeps that true.

## Hardware

Same Mega + RAMPS board as the motion firmware. Load cells (HX711_ADC library):

| Cell | DOUT | SCK | Gripper | Status |
|---|---|---|---|---|
| 1 | A5 | A9 | **GEKKO** | wired, calibrated (655.1 counts/g) |
| 2 | A10 | A11 | **SILICONE** | not yet wired → `LOADCELL_ENABLED[1] = false` |

No pin conflicts: the motion firmware uses 24/26/28/30/34/36/38/54(A0)/55(A1)/56(A2)/60(A6)/61(A7); the cells use A5/A9/A10/A11.

The `LOADCELL_ENABLED[]` guard carries over unchanged — an unwired HX711 has a floating DOUT that makes `update()` do a full 24-bit read on every loop pass, starving AccelStepper.

Baud stays **9600**: the handoff spec pins it, and the HX711 is not a serial device.

## Serial protocol

Extends the existing line protocol. Existing commands are untouched.

| Command | Behavior |
|---|---|
| `FORCE <newtons>` | Store a one-shot target, reply `DONE FORCE` immediately (no motion). `FORCE 0` clears the target. Malformed/negative value → `ERR bad value`. |
| `GRIP CLOSE <mm>` | **No target armed:** position mode, exactly today's behavior. **Target armed:** the mm argument is still parsed and validated (a malformed or negative width is `ERR` exactly as today) but not used for motion; jaws creep closed until the selected gripper's cell reads ≥ target, then Z rises 50 mm, then one `DONE GRIP`. The target clears once consumed. |
| `GRIP OPEN` | Unchanged motion; additionally clears any armed target and aborts an in-progress seek or lift (grip and Z stop where the abort catches them; `DONE GRIP` for the open prints as usual). |
| `F?` | One force line, carried from the load-cell sketch: `F <t_ms> <g1> <N1> <g2> <N2>` (disabled cell reports 0). |
| `TARE` / `TARE <1\|2>` | Non-blocking re-zero via `tareNoDelay()`, result reported when it completes. |
| `CAL <1\|2> <grams>` / `CAL?` | On-rig recalibration and readback, persisted to EEPROM (same `LoadCellCal` struct, magic/version/checksum). |

`FORCE` acks with `DONE FORCE` — not silence — because the sender writes one line and blocks for its ack (handoff spec, "Sender behavior").

### Failure paths

| Condition | Response |
|---|---|
| Jaws reach fully closed (`GRIP_MAX_STEPS`) without hitting the target | `WARN force not reached`, **no lift**, `DONE GRIP`. A missed grasp is a physical outcome, not a protocol violation; the sender collects `WARN`s and never hangs. |
| Force mode commanded while the selected gripper's cell is disabled (silicone today) | `ERR silicone load cell disabled` (or `gekko …`), no motion, no `DONE`. Fatal to the sender by design — the grasp cannot proceed. |
| Cell timeout at boot (`getTareTimeoutFlag`) | `ERR cell <n> timed out …` printed at startup, cell treated as disabled for force mode. |

### Gripper → cell selection

The firmware tracks the last `SELECT` in a `selectedGripper` variable: boot default GEKKO (the turret's assumed boot position is step 0 = gekko), updated when a `SELECT` move is **commanded** (the turret will be there before any `GRIP CLOSE` because the sender waits for `DONE SELECT`). Mapping lives in one constant block next to `SELECT_GEKKO_STEPS` so a head swap is a one-line fix.

## Force-seek control

- **Creep speed:** `FORCE_SEEK_SPEED_STEPS_PER_SEC = 400.0` (2 mm/s of jaw travel). Applied with `setMaxSpeed()` when the seek starts, restored to `GRIP_MAX_SPEED_STEPS_PER_SEC` when the grip axis returns to idle — so a following position-mode move is full speed again.
- **Sensor lag budget:** stock HX711 runs 10 SPS; the library's default 16-sample average is a ~1.6 s window. The merged firmware calls `setSamplesInUse(4)` (~0.4 s window), giving ≈0.8 mm of jaw travel between force-reached and firmware-noticed at creep speed. Acceptable for compliant pads; both knobs are named constants, and tying the HX711 RATE pin high (80 SPS) is the hardware upgrade path.
- **Comparison:** `filteredGrams * G_TO_NEWTON >= targetN`, using the same EMA (`LOADCELL_ALPHA = 0.15`) the load-cell sketch maintains per cell.
- **Motor-side overshoot:** `stop()` decelerates from 400 steps/s at 5000 steps/s² — ~16 steps ≈ 0.08 mm. Negligible next to the sensor lag.
- **Boot:** cells init, stabilize (~2 s), and tare in `setup()` exactly as the standalone sketch does (`startMultiple`, hardcoded-cal path, EEPROM restore for tare offsets). The Python sender already sleeps 2.0 s after opening the port; a command arriving moments early waits in the 64-byte RX buffer.

## State machine

New grip-axis phase enum replacing the bare `gripMovePending` flag:

```
GRIP_IDLE ──GRIP CLOSE (no target)──▶ GRIP_POSITION ──distanceToGo()==0──▶ DONE GRIP, GRIP_IDLE
GRIP_IDLE ──GRIP CLOSE (target)────▶ GRIP_SEEKING
GRIP_SEEKING ──force ≥ target──▶ stop grip, start Z +50 mm ──▶ GRIP_LIFTING
GRIP_SEEKING ──fully closed────▶ WARN force not reached, DONE GRIP, GRIP_IDLE
GRIP_LIFTING ──Z distanceToGo()==0──▶ DONE GRIP, GRIP_IDLE
```

- `GRIP_POSITION` is a rename of today's `gripMovePending == true` — identical observable behavior.
- The lift is Z motion owned by the **grip** operation: it must **not** print `DONE Z` (a stray `DONE Z` would desynchronize the sender's line-by-line ack matching) and must not set `zMovePending`. Lift target = current Z position raised by `LIFT_MM = 50.0`, clamped at top of travel (`Z_MIN_STEPS`).
- `GRIP OPEN` from any phase: clear target, `stop()` grip (and Z if lifting), proceed with the open as a normal `GRIP_POSITION` move.
- `serviceLoadCells()` (the sketch's non-blocking per-pass update, minus SETTLE) runs every `loop()` pass alongside the four `run()` calls. HX711 reads are ~100 µs when a sample is ready — tolerable at Z's 2500 steps/s peak, irrelevant at the 400 steps/s creep.

## Regression guarantee

The user's explicit requirement: the existing serialed commands must not change. Concretely:

1. `Z <mm>`, `SELECT GEKKO|SILICONE`, `GRIP OPEN`, `GRIP CLOSE <mm>` (no target) produce the same motion and the same serial output, in the same order, as the current firmware.
2. The command dispatch stays one `if/else` chain in `processCommand`; new branches (`FORCE`, `F?`, `TARE`, `CAL`, `CAL?`) are appended, and the half-written `FORCE` stub already in the working tree is replaced by the real branch.
3. `parseFloatStrict`, `clampSteps`, `sendErr` are shared, not duplicated — the load-cell code adopts the motion firmware's copies (they are byte-identical between the two sketches).
4. The rig checklist below re-runs the existing commands, not just the new ones.

## Verification

No unit-test harness exists for `.ino`; verification is a compile plus a rig checklist.

**Compile:** `arduino-cli compile` for the Mega with `AccelStepper`, `HX711_ADC`, `EEPROM`. Zero warnings tolerated for narrowing or sign issues in the new code.

**Rig checklist:**

1. Boot: `READY`-style banner after ~2 s, `CAL?` shows cell 1 factor, cell 2 disabled.
2. `F?` reads ~0 g unloaded; press the gekko pad — reading rises; `TARE` re-zeros.
3. Regression: `Z 150`, `SELECT SILICONE`, `SELECT GEKKO`, `GRIP CLOSE 90`, `GRIP OPEN` — same behavior and replies as current firmware.
4. `FORCE 0.5` → `DONE FORCE`; `GRIP CLOSE 60` on a soft object → creep, stop near 0.5 N (check `F?` after), 50 mm lift, single `DONE GRIP`, no `DONE Z`.
5. `FORCE 0.5`, `GRIP CLOSE 60` with nothing between the jaws → full close, `WARN force not reached`, `DONE GRIP`, no lift.
6. `SELECT SILICONE`, `FORCE 0.5`, `GRIP CLOSE 60` → `ERR silicone load cell disabled`, no motion.
7. One-shot check: after a force grasp completes, a bare `GRIP CLOSE 90` is position-mode again.

## Follow-ups (recorded, not in scope)

1. Wire cell 2, flip `LOADCELL_ENABLED[1]`, calibrate — silicone force grasps unlock with no code change.
2. HX711 RATE pin high → 80 SPS → tighter force stop; revisit `setSamplesInUse` and creep speed then.
3. Slip detection during lift (force drop while `GRIP_LIFTING`) if it turns out to matter in practice.
4. Multi-grasp sequencing: re-send `Z` after each lift, or add a `Z` re-home convention.
