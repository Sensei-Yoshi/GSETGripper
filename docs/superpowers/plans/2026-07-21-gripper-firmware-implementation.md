# 4-Motor Gripper Firmware Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `arduino/main/main.ino` from a single Z-axis stepper to 4 steppers (mirrored dual-Z lead screw, gripper-select, gripper open/close) driven by a line-based serial command protocol, with no onboard sequencing.

**Architecture:** One Arduino Mega sketch, 4 `AccelStepper` (STEP/DIR driver mode) instances. The Mega is a pure command executor — it never decides what to do next, only executes whatever line Python sends and reports `DONE <AXIS>` when the move finishes. Shared helpers (`parseFloatStrict`, `clampSteps`, `sendErr`) are introduced once (Task 2) and reused by every later axis.

**Tech Stack:** Arduino (C++), AccelStepper library, arduino-cli for compile verification (no hardware-in-the-loop test rig in this repo).

## Global Constraints

- Serial: line-terminated (`\n`) ASCII commands, 9600 baud (unchanged from current `main.ino`).
- All 4 motors are STEP/DIR driver types, same `AccelStepper::DRIVER` mode as the existing Z motor.
- No onboard sequencing — each serial command triggers exactly one axis move; sequencing (select → lower → close → raise) lives entirely in Python, not in firmware.
- No homing/limit switches on any axis (open-loop from power-on/reset position), but every axis has soft min/max step clamps regardless.
- Exact calibration values (steps/mm, named-position step counts, grip width→steps constants) are unknown — mark with `// TODO: calibrate` exactly as the existing `STEPS_PER_MM` constant does.
- Response protocol is exact-string: `DONE <AXIS>` on completion, `ERR <reason>` on malformed input (no move, no `DONE`), `WARN <axis> clamped to <value>` when a soft limit clamps a target (move still completes, `DONE` still sent).
- Pin assignments: Z side A = STEP 2 / DIR 3 (existing), Z side B = STEP 4 / DIR 5, SELECT = STEP 6 / DIR 7, GRIP = STEP 8 / DIR 9.

---

## Task 1: Arduino compile tooling setup

**Files:**
- None in the repo — this task installs local tooling only.

**Interfaces:**
- Produces: a working `arduino-cli compile --fqbn arduino:avr:mega arduino/main` command that later tasks use as their verification step.

- [ ] **Step 1: Install arduino-cli**

```bash
brew install arduino-cli
arduino-cli version
```
Expected: prints an `arduino-cli` version string (no error).

- [ ] **Step 2: Install the AVR board core**

```bash
arduino-cli core update-index
arduino-cli core install arduino:avr
```
Expected: ends with `arduino:avr@<version> installed` (or already-installed confirmation — this machine already has an AVR core under `~/Library/Arduino15/packages/arduino/hardware/avr`, but `arduino-cli` needs its own index/core install to use it).

- [ ] **Step 3: Install the AccelStepper library**

```bash
arduino-cli lib install AccelStepper
```
Expected: ends with `AccelStepper@<version> installed`.

- [ ] **Step 4: Verify the existing sketch compiles (baseline)**

```bash
arduino-cli compile --fqbn arduino:avr:mega arduino/main
```
Expected: ends with `Sketch uses ... bytes` success output, no errors. This confirms the toolchain works before any code changes — if this fails, fix the toolchain (not the sketch) before proceeding.

No commit for this task — nothing in the repo changes, only local tooling.

---

## Task 2: Response helpers, input validation, soft limits, and DONE for the existing Z axis

Refactors the current single-motor Z handling to the shared conventions every later axis will reuse: strict numeric parsing (so `Z abc` produces `ERR` instead of silently moving to 0), soft-limit clamping with `WARN`, and a `DONE Z` sent once the move actually finishes. Renames the existing `stepper`/`STEP_PIN`/`DIR_PIN`/`STEPS_PER_MM` identifiers to the `Z_`-prefixed names the final 4-axis file uses, so Task 3 (adding the mirrored second Z motor) is a small diff instead of another full rewrite.

**Files:**
- Modify: `arduino/main/main.ino` (full rewrite of the current 61-line file)

**Interfaces:**
- Produces: `bool parseFloatStrict(const String& s, float& out)`, `long clampSteps(long value, long minSteps, long maxSteps, const char* axisName)`, `void sendErr(const char* reason)` — all 3 are reused unmodified by Tasks 4 and 5.
- Produces: `void moveZTo(float targetHeightMM)`, `AccelStepper stepperZA`, `bool zMovePending` — extended (not replaced) by Task 3.

- [ ] **Step 1: Rewrite `arduino/main/main.ino`**

Replace the entire file contents with:

```cpp
#include <AccelStepper.h>

const unsigned long BAUD_RATE = 9600;

// ---- Z axis (lead screw) ----
const int Z_A_STEP_PIN = 2;
const int Z_A_DIR_PIN = 3;
AccelStepper stepperZA(AccelStepper::DRIVER, Z_A_STEP_PIN, Z_A_DIR_PIN);

const float Z_MAX_SPEED_STEPS_PER_SEC = 800.0;
const float Z_ACCELERATION_STEPS_PER_SEC2 = 400.0;

// TODO: calibrate for the real gantry (lead screw pitch, microstepping, etc.).
// Z 0 is wherever the arm physically is when the board powers on/resets --
// there's no homing routine or limit switch yet, so it is not tied to any
// fixed real-world height until one is added.
const float Z_STEPS_PER_MM = 10.0;
const long Z_MIN_STEPS = 0;
const long Z_MAX_STEPS = 5000; // TODO: calibrate safe travel range

bool zMovePending = false;

String command;

void setup() {
  Serial.begin(BAUD_RATE);

  stepperZA.setMaxSpeed(Z_MAX_SPEED_STEPS_PER_SEC);
  stepperZA.setAcceleration(Z_ACCELERATION_STEPS_PER_SEC2);
}

void loop() {
  while (Serial.available() > 0) {
    char incoming = Serial.read();

    if (incoming == '\n') {
      command.trim();
      processCommand(command);
      command = "";
    } else {
      command += incoming;
    }
  }

  // Must be called as often as possible for AccelStepper to step correctly.
  stepperZA.run();

  if (zMovePending && stepperZA.distanceToGo() == 0) {
    zMovePending = false;
    Serial.println("DONE Z");
  }
}

void processCommand(const String& message) {
  if (message.length() == 0) {
    return;
  }

  if (message.startsWith("Z ")) {
    String arg = message.substring(2);
    arg.trim();
    float targetHeightMM;
    if (!parseFloatStrict(arg, targetHeightMM)) {
      sendErr("bad value");
      return;
    }
    moveZTo(targetHeightMM);
  } else {
    sendErr("unknown command");
  }
}

void moveZTo(float targetHeightMM) {
  long targetSteps = clampSteps((long)(targetHeightMM * Z_STEPS_PER_MM), Z_MIN_STEPS, Z_MAX_STEPS, "Z");
  stepperZA.moveTo(targetSteps);
  zMovePending = true;
}

long clampSteps(long value, long minSteps, long maxSteps, const char* axisName) {
  long clamped = value;
  if (clamped < minSteps) {
    clamped = minSteps;
  } else if (clamped > maxSteps) {
    clamped = maxSteps;
  }
  if (clamped != value) {
    Serial.print("WARN ");
    Serial.print(axisName);
    Serial.print(" clamped to ");
    Serial.println(clamped);
  }
  return clamped;
}

bool parseFloatStrict(const String& s, float& out) {
  if (s.length() == 0) {
    return false;
  }
  bool seenDigit = false;
  bool seenDot = false;
  int start = 0;
  if (s[0] == '-' || s[0] == '+') {
    start = 1;
  }
  if (start >= (int)s.length()) {
    return false;
  }
  for (int i = start; i < (int)s.length(); i++) {
    char c = s[i];
    if (c == '.') {
      if (seenDot) {
        return false;
      }
      seenDot = true;
    } else if (isDigit(c)) {
      seenDigit = true;
    } else {
      return false;
    }
  }
  if (!seenDigit) {
    return false;
  }
  out = s.toFloat();
  return true;
}

void sendErr(const char* reason) {
  Serial.print("ERR ");
  Serial.println(reason);
}
```

- [ ] **Step 2: Compile and verify**

```bash
arduino-cli compile --fqbn arduino:avr:mega arduino/main
```
Expected: success output (`Sketch uses ... bytes`), no errors.

- [ ] **Step 3: Commit**

```bash
git add arduino/main/main.ino
git commit -m "Add DONE/ERR/WARN protocol and soft limits to Z axis"
```

---

## Task 3: Mirror the Z axis onto a second motor

Adds `stepperZB` (side B of the lead screw) so both motors always receive identical targets — one logical `Z` command, two physically independent motors moving in lockstep.

**Files:**
- Modify: `arduino/main/main.ino`

**Interfaces:**
- Consumes: `clampSteps`, `zMovePending` from Task 2.
- Produces: `AccelStepper stepperZB` — no later task depends on it directly (Z stays a single logical axis from the protocol's perspective).

- [ ] **Step 1: Add the second Z motor's pins and stepper object**

Find:
```cpp
const int Z_A_STEP_PIN = 2;
const int Z_A_DIR_PIN = 3;
AccelStepper stepperZA(AccelStepper::DRIVER, Z_A_STEP_PIN, Z_A_DIR_PIN);
```
Replace with:
```cpp
const int Z_A_STEP_PIN = 2;
const int Z_A_DIR_PIN = 3;
AccelStepper stepperZA(AccelStepper::DRIVER, Z_A_STEP_PIN, Z_A_DIR_PIN);

const int Z_B_STEP_PIN = 4;
const int Z_B_DIR_PIN = 5;
AccelStepper stepperZB(AccelStepper::DRIVER, Z_B_STEP_PIN, Z_B_DIR_PIN);
```

- [ ] **Step 2: Initialize `stepperZB` in `setup()`**

Find:
```cpp
  stepperZA.setMaxSpeed(Z_MAX_SPEED_STEPS_PER_SEC);
  stepperZA.setAcceleration(Z_ACCELERATION_STEPS_PER_SEC2);
}
```
Replace with:
```cpp
  stepperZA.setMaxSpeed(Z_MAX_SPEED_STEPS_PER_SEC);
  stepperZA.setAcceleration(Z_ACCELERATION_STEPS_PER_SEC2);

  stepperZB.setMaxSpeed(Z_MAX_SPEED_STEPS_PER_SEC);
  stepperZB.setAcceleration(Z_ACCELERATION_STEPS_PER_SEC2);
}
```

- [ ] **Step 3: Run `stepperZB` and require both motors to finish before `DONE Z`**

Find:
```cpp
  // Must be called as often as possible for AccelStepper to step correctly.
  stepperZA.run();

  if (zMovePending && stepperZA.distanceToGo() == 0) {
    zMovePending = false;
    Serial.println("DONE Z");
  }
```
Replace with:
```cpp
  // Must be called as often as possible for AccelStepper to step correctly.
  stepperZA.run();
  stepperZB.run();

  if (zMovePending && stepperZA.distanceToGo() == 0 && stepperZB.distanceToGo() == 0) {
    zMovePending = false;
    Serial.println("DONE Z");
  }
```

- [ ] **Step 4: Mirror the target onto `stepperZB` in `moveZTo`**

Find:
```cpp
void moveZTo(float targetHeightMM) {
  long targetSteps = clampSteps((long)(targetHeightMM * Z_STEPS_PER_MM), Z_MIN_STEPS, Z_MAX_STEPS, "Z");
  stepperZA.moveTo(targetSteps);
  zMovePending = true;
}
```
Replace with:
```cpp
void moveZTo(float targetHeightMM) {
  long targetSteps = clampSteps((long)(targetHeightMM * Z_STEPS_PER_MM), Z_MIN_STEPS, Z_MAX_STEPS, "Z");
  stepperZA.moveTo(targetSteps);
  stepperZB.moveTo(targetSteps);
  zMovePending = true;
}
```

- [ ] **Step 5: Compile and verify**

```bash
arduino-cli compile --fqbn arduino:avr:mega arduino/main
```
Expected: success output, no errors.

- [ ] **Step 6: Commit**

```bash
git add arduino/main/main.ino
git commit -m "Mirror Z axis onto second lead-screw motor"
```

---

## Task 4: Add the SELECT axis (gecko <-> silicone)

**Files:**
- Modify: `arduino/main/main.ino`

**Interfaces:**
- Consumes: `clampSteps`, `sendErr` from Task 2.
- Produces: `void moveSelectTo(long targetSteps)`, `AccelStepper stepperSelect` — not depended on by other tasks (SELECT and GRIP are independent axes).

- [ ] **Step 1: Add SELECT pins, stepper object, and constants**

Find:
```cpp
bool zMovePending = false;

String command;
```
Replace with:
```cpp
bool zMovePending = false;

// ---- SELECT axis (gecko <-> silicone gripper head) ----
const int SELECT_STEP_PIN = 6;
const int SELECT_DIR_PIN = 7;
AccelStepper stepperSelect(AccelStepper::DRIVER, SELECT_STEP_PIN, SELECT_DIR_PIN);

const float SELECT_MAX_SPEED_STEPS_PER_SEC = 400.0;
const float SELECT_ACCELERATION_STEPS_PER_SEC2 = 200.0;

// TODO: calibrate exact step counts for each named position.
const long SELECT_GEKKO_STEPS = 0;
const long SELECT_SILICONE_STEPS = 800;
const long SELECT_MIN_STEPS = 0;
const long SELECT_MAX_STEPS = 800; // TODO: calibrate safe travel range

bool selectMovePending = false;

String command;
```

- [ ] **Step 2: Initialize `stepperSelect` in `setup()`**

Find:
```cpp
  stepperZB.setMaxSpeed(Z_MAX_SPEED_STEPS_PER_SEC);
  stepperZB.setAcceleration(Z_ACCELERATION_STEPS_PER_SEC2);
}
```
Replace with:
```cpp
  stepperZB.setMaxSpeed(Z_MAX_SPEED_STEPS_PER_SEC);
  stepperZB.setAcceleration(Z_ACCELERATION_STEPS_PER_SEC2);

  stepperSelect.setMaxSpeed(SELECT_MAX_SPEED_STEPS_PER_SEC);
  stepperSelect.setAcceleration(SELECT_ACCELERATION_STEPS_PER_SEC2);
}
```

- [ ] **Step 3: Run `stepperSelect` and send `DONE SELECT` on completion**

Find:
```cpp
  if (zMovePending && stepperZA.distanceToGo() == 0 && stepperZB.distanceToGo() == 0) {
    zMovePending = false;
    Serial.println("DONE Z");
  }
```
Replace with:
```cpp
  stepperSelect.run();

  if (zMovePending && stepperZA.distanceToGo() == 0 && stepperZB.distanceToGo() == 0) {
    zMovePending = false;
    Serial.println("DONE Z");
  }
  if (selectMovePending && stepperSelect.distanceToGo() == 0) {
    selectMovePending = false;
    Serial.println("DONE SELECT");
  }
```

- [ ] **Step 4: Handle `SELECT GEKKO` / `SELECT SILICONE` in `processCommand`**

Find:
```cpp
    moveZTo(targetHeightMM);
  } else {
    sendErr("unknown command");
  }
}
```
Replace with:
```cpp
    moveZTo(targetHeightMM);
  } else if (message.startsWith("SELECT ")) {
    String arg = message.substring(7);
    arg.trim();
    if (arg == "GEKKO") {
      moveSelectTo(SELECT_GEKKO_STEPS);
    } else if (arg == "SILICONE") {
      moveSelectTo(SELECT_SILICONE_STEPS);
    } else {
      sendErr("unknown select position");
    }
  } else {
    sendErr("unknown command");
  }
}
```

- [ ] **Step 5: Add `moveSelectTo`**

Find:
```cpp
long clampSteps(long value, long minSteps, long maxSteps, const char* axisName) {
```
Replace with:
```cpp
void moveSelectTo(long targetSteps) {
  long clamped = clampSteps(targetSteps, SELECT_MIN_STEPS, SELECT_MAX_STEPS, "SELECT");
  stepperSelect.moveTo(clamped);
  selectMovePending = true;
}

long clampSteps(long value, long minSteps, long maxSteps, const char* axisName) {
```

- [ ] **Step 6: Compile and verify**

```bash
arduino-cli compile --fqbn arduino:avr:mega arduino/main
```
Expected: success output, no errors.

- [ ] **Step 7: Commit**

```bash
git add arduino/main/main.ino
git commit -m "Add SELECT axis for gecko/silicone gripper head switching"
```

---

## Task 5: Add the GRIP axis (open / variable-width close)

**Files:**
- Modify: `arduino/main/main.ino`

**Interfaces:**
- Consumes: `clampSteps`, `sendErr`, `parseFloatStrict` from Task 2.
- Produces: `void moveGripToSteps(long targetSteps)`, `long gripStepsForWidth(float widthMM)`, `AccelStepper stepperGrip`.

- [ ] **Step 1: Add GRIP pins, stepper object, and constants**

Find:
```cpp
bool selectMovePending = false;

String command;
```
Replace with:
```cpp
bool selectMovePending = false;

// ---- GRIP axis (open/close the active gripper) ----
const int GRIP_STEP_PIN = 8;
const int GRIP_DIR_PIN = 9;
AccelStepper stepperGrip(AccelStepper::DRIVER, GRIP_STEP_PIN, GRIP_DIR_PIN);

const float GRIP_MAX_SPEED_STEPS_PER_SEC = 400.0;
const float GRIP_ACCELERATION_STEPS_PER_SEC2 = 200.0;

// TODO: calibrate. GRIP_OPEN_STEPS is the fully-open home position.
// GRIP_MAX_OPENING_MM is the object width (mm) that corresponds to fully
// open jaws; GRIP_STEPS_PER_MM converts the remaining jaw travel needed to
// close around a narrower object into steps.
const long GRIP_OPEN_STEPS = 0;
const float GRIP_MAX_OPENING_MM = 60.0;
const float GRIP_STEPS_PER_MM = 5.0;
const long GRIP_MIN_STEPS = 0;
const long GRIP_MAX_STEPS = 600; // TODO: calibrate safe travel range

bool gripMovePending = false;

String command;
```

- [ ] **Step 2: Initialize `stepperGrip` in `setup()`**

Find:
```cpp
  stepperSelect.setMaxSpeed(SELECT_MAX_SPEED_STEPS_PER_SEC);
  stepperSelect.setAcceleration(SELECT_ACCELERATION_STEPS_PER_SEC2);
}
```
Replace with:
```cpp
  stepperSelect.setMaxSpeed(SELECT_MAX_SPEED_STEPS_PER_SEC);
  stepperSelect.setAcceleration(SELECT_ACCELERATION_STEPS_PER_SEC2);

  stepperGrip.setMaxSpeed(GRIP_MAX_SPEED_STEPS_PER_SEC);
  stepperGrip.setAcceleration(GRIP_ACCELERATION_STEPS_PER_SEC2);
}
```

- [ ] **Step 3: Run `stepperGrip` and send `DONE GRIP` on completion**

Find:
```cpp
  if (selectMovePending && stepperSelect.distanceToGo() == 0) {
    selectMovePending = false;
    Serial.println("DONE SELECT");
  }
```
Replace with:
```cpp
  stepperGrip.run();

  if (selectMovePending && stepperSelect.distanceToGo() == 0) {
    selectMovePending = false;
    Serial.println("DONE SELECT");
  }
  if (gripMovePending && stepperGrip.distanceToGo() == 0) {
    gripMovePending = false;
    Serial.println("DONE GRIP");
  }
```

- [ ] **Step 4: Handle `GRIP OPEN` / `GRIP CLOSE <mm>` in `processCommand`**

Find:
```cpp
    } else {
      sendErr("unknown select position");
    }
  } else {
    sendErr("unknown command");
  }
}
```
Replace with:
```cpp
    } else {
      sendErr("unknown select position");
    }
  } else if (message.startsWith("GRIP ")) {
    String arg = message.substring(5);
    arg.trim();
    if (arg == "OPEN") {
      moveGripToSteps(GRIP_OPEN_STEPS);
    } else if (arg.startsWith("CLOSE")) {
      String widthArg = arg.substring(5);
      widthArg.trim();
      float widthMM;
      if (!parseFloatStrict(widthArg, widthMM)) {
        sendErr("bad value");
        return;
      }
      moveGripToSteps(gripStepsForWidth(widthMM));
    } else {
      sendErr("unknown grip command");
    }
  } else {
    sendErr("unknown command");
  }
}
```

- [ ] **Step 5: Add `moveGripToSteps` and `gripStepsForWidth`**

Find:
```cpp
long clampSteps(long value, long minSteps, long maxSteps, const char* axisName) {
```
Replace with:
```cpp
void moveGripToSteps(long targetSteps) {
  long clamped = clampSteps(targetSteps, GRIP_MIN_STEPS, GRIP_MAX_STEPS, "GRIP");
  stepperGrip.moveTo(clamped);
  gripMovePending = true;
}

long gripStepsForWidth(float widthMM) {
  float travelMM = GRIP_MAX_OPENING_MM - widthMM;
  if (travelMM < 0) {
    travelMM = 0;
  }
  return GRIP_OPEN_STEPS + (long)(travelMM * GRIP_STEPS_PER_MM);
}

long clampSteps(long value, long minSteps, long maxSteps, const char* axisName) {
```

- [ ] **Step 6: Compile and verify**

```bash
arduino-cli compile --fqbn arduino:avr:mega arduino/main
```
Expected: success output, no errors.

- [ ] **Step 7: Commit**

```bash
git add arduino/main/main.ino
git commit -m "Add GRIP axis with named open position and width-based close"
```

---

## Task 6: Manual bench-test checklist (hardware required)

This task can only be run against real hardware and is not part of the automated compile-verification loop — it's a handoff checklist for whoever has the Mega and motors wired up. Do not mark it complete until each check has actually been run against hardware.

**Files:**
- None (verification only).

- [ ] **Step 1: Flash and open Serial Monitor**

Upload `arduino/main/main.ino` to the Mega, open Serial Monitor at 9600 baud, line ending "Newline".

- [ ] **Step 2: Exercise each axis**

Send each line below and confirm the motor moves correctly and the exact response line appears:

| Send | Expect |
|---|---|
| `Z 50` | `DONE Z` after both Z motors stop moving |
| `SELECT GEKKO` | `DONE SELECT` |
| `SELECT SILICONE` | `DONE SELECT` |
| `GRIP OPEN` | `DONE GRIP` |
| `GRIP CLOSE 40` | `DONE GRIP` |

- [ ] **Step 3: Confirm Z motors stay mechanically mirrored**

While running `Z 50` (or a larger move) under actual load, visually confirm both lead-screw sides move together with no visible racking/binding between the two sides.

- [ ] **Step 4: Exercise error handling**

Send each line below and confirm no motor moves and the exact response appears:

| Send | Expect |
|---|---|
| `Z abc` | `ERR bad value` |
| `GRIP CLOSE` | `ERR bad value` |
| `FOO` | `ERR unknown command` |
| (empty line) | no response |

- [ ] **Step 5: Exercise soft-limit clamping**

Send a value beyond each axis's `_MAX_STEPS`/`_MIN_STEPS` (e.g. `Z 100000` given `Z_MAX_STEPS = 5000` and `Z_STEPS_PER_MM = 10.0`) and confirm a `WARN Z clamped to 5000` line appears, followed by `DONE Z` once the motor reaches the clamped position (not the requested one).

No commit for this task — it's a verification checklist, not a code change.

---

## Self-Review Notes

- **Spec coverage:** Architecture (Task 2/3 pin table + AccelStepper setup), serial protocol table (Tasks 2/4/5 cover `Z`, `SELECT GEKKO/SILICONE`, `GRIP OPEN`, `GRIP CLOSE <mm>`), error handling (`ERR`/`WARN` in Task 2, exercised in Task 6), testing plan (Task 6 mirrors the spec's 4-point manual checklist exactly). No gaps found.
- **Placeholder scan:** No TBD/TODO-without-value in any step; the `// TODO: calibrate` comments are intentional per the spec's explicit non-goal (real calibration values aren't available yet) and match the existing codebase's convention.
- **Type consistency:** `moveZTo(float)`, `moveSelectTo(long)`, `moveGripToSteps(long)`, `gripStepsForWidth(float) -> long`, `clampSteps(long, long, long, const char*) -> long`, `parseFloatStrict(const String&, float&) -> bool`, `sendErr(const char*)` are used with matching signatures everywhere they're called across Tasks 2–5.
