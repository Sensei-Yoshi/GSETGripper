# Force-Controlled Grasp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the HX711 load cells into `arduino/main/main.ino` and implement `FORCE <newtons>` so an armed `GRIP CLOSE` creeps shut until the selected gripper's cell reads the target, lifts 50 mm, and acks with a single `DONE GRIP`.

**Architecture:** Everything lands in the single `main.ino` sketch (repo convention — one sketch per directory, no headers). The load-cell reading path is transplanted from the standalone sketch (non-blocking `update()` per loop pass, EMA filter, EEPROM-persisted calibration). The grip axis's `gripMovePending` flag becomes a four-state phase enum; force-seeking and the automatic lift are new states serviced in `loop()`, never blocking serial or the other axes.

**Tech Stack:** Arduino (C++), AccelStepper, HX711_ADC, EEPROM, `arduino-cli` for compile verification (no hardware-in-the-loop test rig in this repo).

**Spec:** `docs/superpowers/specs/2026-07-27-force-grasp-design.md`

## Global Constraints

- **Zero behavior change to existing commands.** `Z <mm>`, `SELECT GEKKO|SILICONE`, `GRIP OPEN`, `GRIP CLOSE <mm>` (no force target armed) must produce the same motion and the same serial lines, in the same order, as the current firmware. Do not edit their code paths except where a task explicitly says so.
- Baud stays `9600` (`BAUD_RATE`, `main.ino:3`). Do not change it — the Python handoff spec pins it.
- The 50 mm lift must **never** print `DONE Z` or set `zMovePending` — it belongs to the grip operation and a stray `DONE Z` desynchronizes the Python sender's line-by-line ack matching.
- New `Serial.print` strings use the `F()` macro (flash, not RAM), matching the load-cell sketch they come from. Existing prints stay exactly as they are.
- Verification command for every task: `arduino-cli compile --fqbn arduino:avr:mega arduino/main` — must end in success (`Sketch uses N bytes…`). Treat any new warning about narrowing or sign conversion in changed code as a failure.
- Cell 1 (index 0) = GEKKO, cell 2 (index 1) = SILICONE. Cell 2 is unwired: `LOADCELL_ENABLED[1] = false`.

---

### Task 1: Toolchain + baseline commit

**Files:**
- Modify: none (library install + committing the already-modified `arduino/main/main.ino`)

**Interfaces:**
- Consumes: the working tree's pending `main.ino` edits (measured grip constants, `GRIP_CLOSE_SIGN`, negative-width guard, empty `FORCE` stub).
- Produces: a compiling, committed baseline that every later task diffs against. The `HX711_ADC` library available to `arduino-cli`.

- [ ] **Step 1: Install the HX711_ADC library**

```bash
arduino-cli lib install HX711_ADC
```

Expected: ends with `HX711_ADC@<version> installed` (AccelStepper 1.64 and the `arduino:avr` 1.8.8 core are already installed on this machine).

- [ ] **Step 2: Compile the baseline**

Run: `arduino-cli compile --fqbn arduino:avr:mega arduino/main`
Expected: success. This proves any later failure comes from new work, not the pending edits.

- [ ] **Step 3: Commit the pending main.ino edits**

The working tree already carries measured grip-travel constants, `GRIP_CLOSE_SIGN`, the negative-width guard, and a half-written `FORCE` stub (`main.ino:254-257`). Commit them as the baseline; the stub is replaced in Task 4. Do **not** stage `docs/superpowers/plans/2026-07-27-serial-handoff.md` — that diff belongs to the serial-handoff work stream.

```bash
git add arduino/main/main.ino
git commit -m "feat: measured grip travel constants, close sign, negative-width guard"
```

---

### Task 2: Load-cell reading infrastructure

**Files:**
- Modify: `arduino/main/main.ino` — includes at top; new constants block before `String command;` (~line 144); `setup()` additions after `stepperSelect.setPinsInverted(...)` (~line 170); `serviceLoadCells()` call in `loop()`; new functions after `sendErr`.

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces (used by Tasks 3, 4, 6): `void serviceLoadCells()`, `float cellForceN(int i)`, `CellRuntime cellRT[LOADCELL_COUNT]` (fields `float filtered; bool haveFirst; bool tarePending;`), `const bool LOADCELL_ENABLED[2]`, `bool cellHealthy[2]`, `const int CELL_FOR_GEKKO = 0`, `const int CELL_FOR_SILICONE = 1`, `const float G_TO_NEWTON`, `LoadCellCal cal`, `void calSave()`, `void calPrint()`, `void startTare(int i)`, `void calibrateCell(int i, float knownGrams)`, `int cellIndexFromArg(String arg)`, `void printForceLine()`.

- [ ] **Step 1: Add includes**

At the top of `main.ino`, after `#include <AccelStepper.h>`:

```cpp
#include <HX711_ADC.h>
#include <EEPROM.h>
```

- [ ] **Step 2: Add the load-cell constants block**

Insert immediately before `String command;` (~line 144):

```cpp
// ---- HX711 LOAD CELLS (HX711_ADC) ----
// Cell 1 (index 0): DOUT -> A5,  SCK -> A9   -- GEKKO head
// Cell 2 (index 1): DOUT -> A10, SCK -> A11  -- SILICONE head
const int LC1_DOUT = A5;
const int LC1_SCK  = A9;
const int LC2_DOUT = A10;
const int LC2_SCK  = A11;

HX711_ADC loadCell1(LC1_DOUT, LC1_SCK);
HX711_ADC loadCell2(LC2_DOUT, LC2_SCK);

const int LOADCELL_COUNT = 2;
HX711_ADC* loadCell[LOADCELL_COUNT] = {&loadCell1, &loadCell2};

// Which cell the force loop reads for each turret head. Swap these two if the
// heads turn out to be the other way round -- same knob as SELECT_GEKKO_STEPS.
const int CELL_FOR_GEKKO = 0;
const int CELL_FOR_SILICONE = 1;

// Set an entry false for any cell that is not physically wired yet. An unwired
// HX711 leaves DOUT floating; a floating pin reads LOW much of the time, so
// update() believes a conversion is waiting on EVERY loop pass and does a full
// 24-bit shift-out each time. That starves AccelStepper.
const bool LOADCELL_ENABLED[LOADCELL_COUNT] = {true, false};

// Flip the sign properly rather than fabs()-ing the reading. fabs() rectifies
// the noise around zero into a positive bias, which corrupts small-force
// readings and the tared baseline.
const bool LOADCELL_REVERSED[LOADCELL_COUNT] = {false, false};

const unsigned long LOADCELL_STABILIZING_MS = 2000;

// Extra smoothing on top of the library's own moving average. 1.0 = none.
const float LOADCELL_ALPHA = 0.15;

const float G_TO_NEWTON = 0.00980665f; // 1 gram-force -> N

// getData() averages this many samples (highest and lowest discarded). At the
// stock 10 SPS this is a ~0.4 s window -- the force-seek's dominant lag
// (~0.8 mm of jaw travel at creep speed). Tie the HX711 RATE pin high for
// 80 SPS if that proves too coarse.
const int LOADCELL_SAMPLES_IN_USE = 4;

struct CellRuntime {
  float filtered;   // EMA-smoothed grams, relative to tare
  bool haveFirst;
  bool tarePending;
};
CellRuntime cellRT[LOADCELL_COUNT];

// False when a cell timed out at boot -- force mode refuses to use it.
bool cellHealthy[LOADCELL_COUNT] = {true, true};

// ---- EEPROM-persisted calibration ----
const uint32_t CAL_MAGIC = 0x4C43414CUL; // "LCAL"
const uint16_t CAL_VERSION = 2;
const int CAL_EEPROM_ADDR = 0;

struct LoadCellCal {
  uint32_t magic;
  uint16_t version;
  float calFactor[LOADCELL_COUNT]; // raw counts per gram
  long tareOffset[LOADCELL_COUNT]; // raw counts at zero load
  uint16_t checksum; // must stay the LAST member
};
LoadCellCal cal;
bool calValid = false;

// Calibration factor per cell, in raw counts per gram. Cell 1 measured 655.1.
// Use a NEGATIVE value if a cell reads backwards under load.
const float CAL_FACTOR[LOADCELL_COUNT] = {655.1, 655.1};

// true  = always use CAL_FACTOR above and tare fresh on every boot
// false = restore whatever TARE/CAL last saved to EEPROM
const bool USE_HARDCODED_CAL = true;
```

- [ ] **Step 3: Add cell init to `setup()`**

Append after `stepperSelect.setPinsInverted(true, false, false);`:

```cpp
  // ---- load cells ----
  for (int i = 0; i < LOADCELL_COUNT; i++) {
    cellRT[i].filtered = 0.0;
    cellRT[i].haveFirst = false;
    cellRT[i].tarePending = false;

    if (!LOADCELL_ENABLED[i]) {
      Serial.print(F("INFO cell "));
      Serial.print(i + 1);
      Serial.println(F(" disabled in LOADCELL_ENABLED -- not initialized"));
      continue;
    }
    loadCell[i]->begin();
    if (LOADCELL_REVERSED[i]) {
      loadCell[i]->setReverseOutput();
    }
  }

  calLoad();

  // With USE_HARDCODED_CAL we always tare fresh at boot. Otherwise only tare
  // when there is no stored record to restore.
  bool doTare = USE_HARDCODED_CAL || !calValid;

  // A disabled cell counts as already ready so the handshake does not hang.
  byte ready1 = LOADCELL_ENABLED[0] ? 0 : 1;
  byte ready2 = LOADCELL_ENABLED[1] ? 0 : 1;
  while ((ready1 + ready2) < 2) {
    if (!ready1) ready1 = loadCell1.startMultiple(LOADCELL_STABILIZING_MS, doTare);
    if (!ready2) ready2 = loadCell2.startMultiple(LOADCELL_STABILIZING_MS, doTare);
  }

  for (int i = 0; i < LOADCELL_COUNT; i++) {
    if (!LOADCELL_ENABLED[i]) continue;

    if (loadCell[i]->getTareTimeoutFlag() || loadCell[i]->getSignalTimeoutFlag()) {
      cellHealthy[i] = false;
      Serial.print(F("ERR cell "));
      Serial.print(i + 1);
      Serial.println(F(" timed out -- check DOUT/SCK wiring and 5V"));
    }
    if (USE_HARDCODED_CAL) {
      cal.calFactor[i] = CAL_FACTOR[i];
    }
    loadCell[i]->setCalFactor(cal.calFactor[i]);
    loadCell[i]->setSamplesInUse(LOADCELL_SAMPLES_IN_USE);
  }

  if (doTare) {
    for (int i = 0; i < LOADCELL_COUNT; i++) {
      if (!LOADCELL_ENABLED[i]) continue;
      cal.tareOffset[i] = loadCell[i]->getTareOffset();
    }
    if (!USE_HARDCODED_CAL) {
      calSave();
    }
  } else {
    for (int i = 0; i < LOADCELL_COUNT; i++) {
      if (!LOADCELL_ENABLED[i]) continue;
      loadCell[i]->setTareOffset(cal.tareOffset[i]);
    }
  }

  Serial.println(F("READY"));
```

Note: `setup()` now blocks ~2 s for cell stabilization and prints boot lines where the firmware used to print nothing. The Python sender already sleeps 2.0 s after opening the port; Task 7 records the input-buffer-flush note for the sender.

- [ ] **Step 4: Service the cells from `loop()`**

After the four `stepperX.run();` calls in `loop()`, add:

```cpp
  serviceLoadCells();
```

- [ ] **Step 5: Add the load-cell functions**

Append after `sendErr` at the end of the file:

```cpp
// ---------------- LOAD CELLS ----------------

// Non-blocking per-pass update: pick up a new sample when one is ready and
// fold it into the EMA. Also completes any pending tare.
void serviceLoadCells() {
  for (int i = 0; i < LOADCELL_COUNT; i++) {
    if (!LOADCELL_ENABLED[i]) continue;

    if (loadCell[i]->update()) {
      float g = loadCell[i]->getData();
      CellRuntime& rt = cellRT[i];
      if (!rt.haveFirst) {
        rt.filtered = g;
        rt.haveFirst = true;
      } else {
        rt.filtered += LOADCELL_ALPHA * (g - rt.filtered);
      }
    }

    if (cellRT[i].tarePending && loadCell[i]->getTareStatus()) {
      cellRT[i].tarePending = false;
      cal.tareOffset[i] = loadCell[i]->getTareOffset();
      cellRT[i].haveFirst = false;
      Serial.print(F("TARE "));
      Serial.print(i + 1);
      Serial.print(F(" offset="));
      Serial.println(cal.tareOffset[i]);
      if (!USE_HARDCODED_CAL) {
        calSave();
      }
    }
  }
}

float cellForceN(int i) {
  return cellRT[i].filtered * G_TO_NEWTON;
}

// F <t_ms> <g1> <N1> <g2> <N2>. A disabled cell reports 0.
void printForceLine() {
  Serial.print(F("F "));
  Serial.print(millis());
  for (int i = 0; i < LOADCELL_COUNT; i++) {
    Serial.print(' ');
    Serial.print(cellRT[i].filtered, 2);
    Serial.print(' ');
    Serial.print(cellForceN(i), 4);
  }
  Serial.println();
}

int cellIndexFromArg(String arg) {
  arg.trim();
  int idx = -1;
  if (arg == "1") idx = 0;
  if (arg == "2") idx = 1;
  if (idx < 0) {
    sendErr("cell must be 1 or 2");
    return -1;
  }
  if (!LOADCELL_ENABLED[idx]) {
    sendErr("that cell is disabled in LOADCELL_ENABLED");
    return -1;
  }
  return idx;
}

// Non-blocking: the library fills a fresh dataset in the background and the
// result gets picked up in serviceLoadCells().
void startTare(int i) {
  loadCell[i]->tareNoDelay();
  cellRT[i].tarePending = true;
  Serial.print(F("TARING cell "));
  Serial.print(i + 1);
  Serial.println(F(" -- keep it unloaded and still"));
}

void calibrateCell(int i, float knownGrams) {
  if (knownGrams == 0.0) {
    sendErr("known mass cannot be zero");
    return;
  }
  if (cellRT[i].tarePending) {
    sendErr("tare still running");
    return;
  }
  // getNewCalibration() both computes and applies the factor from the current
  // averaged reading, so the mass must already be on and settled.
  float newFactor = loadCell[i]->getNewCalibration(knownGrams);
  cal.calFactor[i] = newFactor;
  cellRT[i].haveFirst = false;
  Serial.print(F("CAL "));
  Serial.print(i + 1);
  Serial.print(F(" calFactor="));
  Serial.println(newFactor, 4);
  if (USE_HARDCODED_CAL) {
    Serial.println(F("NOTE paste that number into CAL_FACTOR[] -- it resets at next boot"));
  } else {
    calSave();
  }
}

// ---------------- EEPROM CALIBRATION STORAGE ----------------

uint16_t calChecksum(const LoadCellCal& c) {
  const uint8_t* p = (const uint8_t*)&c;
  size_t n = sizeof(LoadCellCal) - sizeof(c.checksum);
  uint16_t sum = 0;
  for (size_t k = 0; k < n; k++) {
    sum += p[k];
  }
  return sum;
}

void calSetDefaults() {
  memset(&cal, 0, sizeof(cal));
  cal.magic = CAL_MAGIC;
  cal.version = CAL_VERSION;
  for (int i = 0; i < LOADCELL_COUNT; i++) {
    cal.calFactor[i] = CAL_FACTOR[i];
    cal.tareOffset[i] = 0;
  }
  calValid = false;
}

void calLoad() {
  LoadCellCal tmp;
  EEPROM.get(CAL_EEPROM_ADDR, tmp);
  if (tmp.magic == CAL_MAGIC && tmp.version == CAL_VERSION &&
      tmp.checksum == calChecksum(tmp)) {
    cal = tmp;
    calValid = true;
  } else {
    calSetDefaults();
  }
}

void calSave() {
  cal.magic = CAL_MAGIC;
  cal.version = CAL_VERSION;
  cal.checksum = calChecksum(cal);
  EEPROM.put(CAL_EEPROM_ADDR, cal);
  calValid = true;
  Serial.println(F("CAL SAVED"));
}

void calPrint() {
  for (int i = 0; i < LOADCELL_COUNT; i++) {
    Serial.print(F("CAL "));
    Serial.print(i + 1);
    if (!LOADCELL_ENABLED[i]) {
      Serial.println(F(" disabled"));
      continue;
    }
    Serial.print(F(" calFactor="));
    Serial.print(cal.calFactor[i], 4);
    Serial.print(F(" tareOffset="));
    Serial.println(cal.tareOffset[i]);
  }
}
```

Semantics note (spec'd): `USE_HARDCODED_CAL` stays `true`, exactly like the standalone sketch — boot uses `CAL_FACTOR[]` and tares fresh; `CAL` prints the new factor with a NOTE instead of persisting. Flipping the flag to `false` enables the EEPROM restore path with no other change.

- [ ] **Step 6: Compile**

Run: `arduino-cli compile --fqbn arduino:avr:mega arduino/main`
Expected: success, no new warnings.

- [ ] **Step 7: Commit**

```bash
git add arduino/main/main.ino
git commit -m "feat: integrate HX711 load cell reading path into main firmware"
```

---

### Task 3: Load-cell utility commands (`F?`, `TARE`, `CAL`, `CAL?`)

**Files:**
- Modify: `arduino/main/main.ino` — `processCommand` only: new `else if` branches inserted between the `FORCE` stub and the final `else`.

**Interfaces:**
- Consumes (Task 2): `printForceLine()`, `startTare(i)`, `cellIndexFromArg(arg)`, `calibrateCell(i, g)`, `calPrint()`, `parseFloatStrict`, `LOADCELL_ENABLED`, `LOADCELL_COUNT`.
- Produces: the four utility commands on the wire. No later task depends on them.

- [ ] **Step 1: Add the dispatch branches**

In `processCommand`, immediately before the final `else {`, add:

```cpp
  else if (message == "F?") {
    printForceLine();

  } else if (message == "TARE") {
    for (int i = 0; i < LOADCELL_COUNT; i++) {
      if (!LOADCELL_ENABLED[i]) continue;
      startTare(i);
    }

  } else if (message.startsWith("TARE ")) {
    int idx = cellIndexFromArg(message.substring(5));
    if (idx < 0) return;
    startTare(idx);

  } else if (message.startsWith("CAL ")) {
    String arg = message.substring(4);
    arg.trim();
    int sp = arg.indexOf(' ');
    if (sp < 0) {
      sendErr("usage: CAL <1|2> <known grams>");
      return;
    }
    int idx = cellIndexFromArg(arg.substring(0, sp));
    if (idx < 0) return;
    float grams;
    if (!parseFloatStrict(arg.substring(sp + 1), grams)) {
      sendErr("bad value");
      return;
    }
    calibrateCell(idx, grams);

  } else if (message == "CAL?") {
    calPrint();
  }
```

Do not touch the existing `Z` / `SELECT` / `GRIP` branches.

- [ ] **Step 2: Compile**

Run: `arduino-cli compile --fqbn arduino:avr:mega arduino/main`
Expected: success.

- [ ] **Step 3: Commit**

```bash
git add arduino/main/main.ino
git commit -m "feat: add F?, TARE, CAL, CAL? load-cell commands"
```

---

### Task 4: `FORCE` command + selected-gripper tracking

**Files:**
- Modify: `arduino/main/main.ino` — globals near the grip constants; the `SELECT` branch; replace the `FORCE` stub in `processCommand`.

**Interfaces:**
- Consumes (Task 2): `CELL_FOR_GEKKO`, `CELL_FOR_SILICONE`.
- Produces (used by Task 6): `bool forceArmed`, `float forceTargetN`, `int selectedCellIndex`.

- [ ] **Step 1: Add the target and selection globals**

Insert immediately before `String command;`:

```cpp
// ---- FORCE target (one-shot) ----
// Armed by "FORCE <n>", consumed by the next "GRIP CLOSE", cleared by
// "FORCE 0" and "GRIP OPEN". While unarmed, GRIP CLOSE is pure position mode.
bool forceArmed = false;
float forceTargetN = 0.0;

// Which cell the next force grasp reads: follows the last SELECT command.
// Boot default GEKKO because the turret's assumed boot position is step 0.
int selectedCellIndex = CELL_FOR_GEKKO;
```

- [ ] **Step 2: Track selection in the `SELECT` branch**

Change only the two bodies inside the existing branch (the `moveSelectTo` calls and error handling stay identical):

```cpp
    if (arg == "GEKKO") {
      selectedCellIndex = CELL_FOR_GEKKO;
      moveSelectTo(SELECT_GEKKO_STEPS);
    } else if (arg == "SILICONE") {
      selectedCellIndex = CELL_FOR_SILICONE;
      moveSelectTo(SELECT_SILICONE_STEPS);
    } else {
      sendErr("unknown select position");
    }
```

- [ ] **Step 3: Replace the `FORCE` stub**

Delete the current empty stub:

```cpp
  else if (message.startsWith("FORCE")){
    String arg = message.substring(6);
    arg.trim();
  }
```

and put in its place (note the trailing space in the prefix — a bare `FORCE` falls through to `ERR unknown command`):

```cpp
  else if (message.startsWith("FORCE ")) {
    String arg = message.substring(6);
    arg.trim();
    float newtons;
    if (!parseFloatStrict(arg, newtons) || newtons < 0.0) {
      sendErr("bad value");
      return;
    }
    if (newtons == 0.0) {
      forceArmed = false;  // FORCE 0 returns GRIP CLOSE to position mode
    } else {
      forceTargetN = newtons;
      forceArmed = true;
    }
    Serial.println("DONE FORCE");
  }
```

`DONE FORCE` prints in both arm and clear cases — the Python sender blocks for an ack after every line it writes.

- [ ] **Step 4: Compile**

Run: `arduino-cli compile --fqbn arduino:avr:mega arduino/main`
Expected: success. (Until Task 6, an armed target is stored but `GRIP CLOSE` still position-moves — acceptable mid-branch state, flagged here so nobody flashes this commit expecting force control.)

- [ ] **Step 5: Commit**

```bash
git add arduino/main/main.ino
git commit -m "feat: FORCE command stores one-shot target; SELECT tracks active cell"
```

---

### Task 5: Grip phase enum (pure refactor, no behavior change)

**Files:**
- Modify: `arduino/main/main.ino` — replace `bool gripMovePending` (~line 142), its assignment in `moveGripToSteps`, and its check in `loop()`.

**Interfaces:**
- Consumes: nothing new.
- Produces (used by Task 6): `enum GripPhase { GRIP_IDLE, GRIP_POSITION, GRIP_SEEKING, GRIP_LIFTING }; GripPhase gripPhase;` — with `GRIP_POSITION` behaving exactly as `gripMovePending == true` does today.

- [ ] **Step 1: Replace the flag with the enum**

Replace `bool gripMovePending = false;` with:

```cpp
// Grip-axis lifecycle. GRIP_POSITION is the classic position move (what the
// old gripMovePending flag tracked). GRIP_SEEKING and GRIP_LIFTING are the
// force-controlled grasp, serviced in loop().
enum GripPhase { GRIP_IDLE, GRIP_POSITION, GRIP_SEEKING, GRIP_LIFTING };
GripPhase gripPhase = GRIP_IDLE;
```

- [ ] **Step 2: Update `moveGripToSteps`**

Replace `gripMovePending = true;` with:

```cpp
  gripPhase = GRIP_POSITION;
```

- [ ] **Step 3: Update the completion check in `loop()`**

Replace:

```cpp
  if (gripMovePending && stepperGrip.distanceToGo() == 0) {
    gripMovePending = false;
    Serial.println("DONE GRIP");
  }
```

with:

```cpp
  if (gripPhase == GRIP_POSITION && stepperGrip.distanceToGo() == 0) {
    gripPhase = GRIP_IDLE;
    Serial.println("DONE GRIP");
  }
```

- [ ] **Step 4: Compile and check no other references remain**

Run: `arduino-cli compile --fqbn arduino:avr:mega arduino/main && ! grep -n gripMovePending arduino/main/main.ino`
Expected: compile success and no `gripMovePending` occurrences.

- [ ] **Step 5: Commit**

```bash
git add arduino/main/main.ino
git commit -m "refactor: grip axis phase enum in place of gripMovePending"
```

---

### Task 6: Force-seek, automatic 50 mm lift, and abort

**Files:**
- Modify: `arduino/main/main.ino` — force-grasp constants near the grip constants; `GRIP CLOSE` and `GRIP OPEN` branches; the grip phase handling in `loop()`; new functions `startForceSeek()` and `startLift()`.

**Interfaces:**
- Consumes: Task 2 (`cellRT`, `cellHealthy`, `LOADCELL_ENABLED`, `cellForceN`, `CELL_FOR_*`), Task 4 (`forceArmed`, `forceTargetN`, `selectedCellIndex`), Task 5 (`gripPhase`, `GripPhase` values).
- Produces: the complete force-grasp behavior. No later task.

- [ ] **Step 1: Add the force-grasp constants and seek state**

Insert immediately after the `forceArmed` / `selectedCellIndex` block from Task 4:

```cpp
// Creep speed while seeking force contact -- 2 mm/s of jaw travel. Restored
// to GRIP_MAX_SPEED_STEPS_PER_SEC when the grip axis returns to idle.
const float FORCE_SEEK_SPEED_STEPS_PER_SEC = 400.0;

// Rise after a successful force grasp, before DONE GRIP is sent.
const float LIFT_MM = 50.0;

// Consumed copies for the seek in progress (forceArmed is one-shot).
int activeCellIndex = 0;
float activeTargetN = 0.0;
```

- [ ] **Step 2: Branch `GRIP CLOSE` on the armed target**

In the `GRIP CLOSE` branch, replace only the final motion line

```cpp
      moveGripToSteps(gripStepsForWidth(widthMM));
```

with:

```cpp
      if (forceArmed) {
        // Force mode: widthMM was validated above but is not used for motion.
        startForceSeek();
      } else {
        moveGripToSteps(gripStepsForWidth(widthMM));
      }
```

The parse and negative-width guard above it stay exactly as they are — a malformed width is `ERR bad value` in both modes.

- [ ] **Step 3: Make `GRIP OPEN` clear and abort**

Replace the `GRIP OPEN` body:

```cpp
    if (arg == "OPEN") {
      moveGripToSteps(GRIP_OPEN_STEPS);
    }
```

with:

```cpp
    if (arg == "OPEN") {
      // Clears any armed target and aborts an in-progress seek or lift. The
      // lift's Z motion stops where the abort catches it (it set no
      // zMovePending, so stopping it prints nothing).
      forceArmed = false;
      if (gripPhase == GRIP_LIFTING) {
        stepperZA.stop();
        stepperZB.stop();
      }
      stepperGrip.setMaxSpeed(GRIP_MAX_SPEED_STEPS_PER_SEC);
      moveGripToSteps(GRIP_OPEN_STEPS);
    }
```

- [ ] **Step 4: Extend the phase handling in `loop()`**

Replace the Task 5 block:

```cpp
  if (gripPhase == GRIP_POSITION && stepperGrip.distanceToGo() == 0) {
    gripPhase = GRIP_IDLE;
    Serial.println("DONE GRIP");
  }
```

with:

```cpp
  if (gripPhase == GRIP_POSITION && stepperGrip.distanceToGo() == 0) {
    gripPhase = GRIP_IDLE;
    Serial.println("DONE GRIP");
  } else if (gripPhase == GRIP_SEEKING) {
    if (cellRT[activeCellIndex].haveFirst &&
        cellForceN(activeCellIndex) >= activeTargetN) {
      // Contact at target force: stop the jaws (decelerate; the lead screw
      // holds from there) and lift the object.
      stepperGrip.stop();
      startLift();
      gripPhase = GRIP_LIFTING;
    } else if (stepperGrip.distanceToGo() == 0) {
      // Fully closed without reaching the target: missed or slipped object.
      // A physical outcome, not a protocol violation -- WARN, no lift, DONE.
      stepperGrip.setMaxSpeed(GRIP_MAX_SPEED_STEPS_PER_SEC);
      Serial.println("WARN force not reached");
      gripPhase = GRIP_IDLE;
      Serial.println("DONE GRIP");
    }
  } else if (gripPhase == GRIP_LIFTING) {
    if (stepperZA.distanceToGo() == 0 && stepperZB.distanceToGo() == 0) {
      stepperGrip.setMaxSpeed(GRIP_MAX_SPEED_STEPS_PER_SEC);
      gripPhase = GRIP_IDLE;
      Serial.println("DONE GRIP");
    }
  }
```

- [ ] **Step 5: Add `startForceSeek()` and `startLift()`**

After `moveGripToSteps`:

```cpp
// Begin a force-seeking close: creep toward fully closed and let loop() stop
// the jaws when the selected gripper's cell reads the target. Refuses (ERR,
// no DONE) when that cell cannot be read -- the grasp cannot proceed, and the
// Python sender treats ERR as fatal by design.
void startForceSeek() {
  int cell = selectedCellIndex;
  if (!LOADCELL_ENABLED[cell] || !cellHealthy[cell]) {
    sendErr(cell == CELL_FOR_GEKKO ? "gekko load cell disabled"
                                   : "silicone load cell disabled");
    return;
  }
  forceArmed = false;  // one-shot: consumed by this close
  activeCellIndex = cell;
  activeTargetN = forceTargetN;
  stepperGrip.setMaxSpeed(FORCE_SEEK_SPEED_STEPS_PER_SEC);
  stepperGrip.moveTo(GRIP_CLOSE_SIGN * GRIP_MAX_STEPS);
  gripPhase = GRIP_SEEKING;
}

// Raise Z by LIFT_MM from wherever it is, clamped at top of travel. Owned by
// the grip operation: deliberately does NOT set zMovePending, so it never
// prints DONE Z -- the sender is waiting for DONE GRIP.
void startLift() {
  long liftSteps = (long)(LIFT_MM * Z_STEPS_PER_MM);
  // Signed motor position -> non-negative descent steps from top-of-travel.
  long descentSteps = Z_DOWN_SIGN * stepperZA.currentPosition();
  long targetDescent = descentSteps - liftSteps;
  if (targetDescent < Z_MIN_STEPS) {
    targetDescent = Z_MIN_STEPS;
  }
  stepperZA.moveTo(Z_DOWN_SIGN * targetDescent);
  stepperZB.moveTo(Z_DOWN_SIGN * targetDescent);
}
```

- [ ] **Step 6: Compile**

Run: `arduino-cli compile --fqbn arduino:avr:mega arduino/main`
Expected: success, no new warnings.

- [ ] **Step 7: Commit**

```bash
git add arduino/main/main.ino
git commit -m "feat: force-seeking grasp with automatic 50 mm lift"
```

---

### Task 7: Rig verification + sender note

**Files:**
- Modify: `docs/superpowers/plans/2026-07-27-serial-handoff.md` (one appended note)
- Manual: the physical rig, over a serial monitor at 9600 baud

**Interfaces:**
- Consumes: everything above, flashed to the Mega.
- Produces: verified firmware; a recorded Python-side follow-up.

- [ ] **Step 1: Record the boot-output note for the serial-handoff sender**

The firmware now prints boot lines (`INFO …`, `READY`, possibly `ERR cell … timed out`) where it used to print nothing. Append this note at the end of `docs/superpowers/plans/2026-07-27-serial-handoff.md`:

```markdown
## Note from the force-grasp firmware work (2026-07-27)

The merged firmware prints boot lines (`INFO`/`READY`, and `ERR cell <n> timed
out` on a wiring fault) before accepting commands. `SerialGraspSender` must
call `reset_input_buffer()` after its 2.0 s post-open sleep, or the first
`Z` ack read will consume boot output — and a boot `ERR` line would be
mistaken for a command failure.
```

```bash
git add docs/superpowers/plans/2026-07-27-serial-handoff.md
git commit -m "docs: sender must flush boot output before first command"
```

- [ ] **Step 2: Flash**

With the carriage parked at top of travel and the jaws fully open (the no-homing precondition):

```bash
arduino-cli upload --fqbn arduino:avr:mega -p <port> arduino/main
```

(`arduino-cli board list` shows the port.)

- [ ] **Step 3: Run the rig checklist (from the spec)**

Over a serial monitor at 9600 baud, verify each line — check off only what passes:

1. Boot: `READY` after ~2 s; `CAL?` shows cell 1's factor and `CAL 2 disabled`.
2. `F?` reads ~0 g unloaded; pressing the gekko pad raises it; `TARE` re-zeros (prints `TARING…` then `TARE 1 offset=…`).
3. **Regression:** `Z 150`, `SELECT SILICONE`, `SELECT GEKKO`, `GRIP CLOSE 90`, `GRIP OPEN` — motion and replies identical to the pre-change firmware (`DONE Z`, `DONE SELECT` ×2, `DONE GRIP` ×2, plus the usual WARN behavior on clamped values).
4. `FORCE 0.5` → `DONE FORCE`; `GRIP CLOSE 60` on a soft object → slow creep, jaws stop near 0.5 N (`F?` afterward), 50 mm rise, then exactly one `DONE GRIP` and **no** `DONE Z`.
5. `FORCE 0.5`; `GRIP CLOSE 60` with nothing between the jaws → full slow close, `WARN force not reached`, `DONE GRIP`, no lift.
6. `SELECT SILICONE` → `DONE SELECT`; `FORCE 0.5` → `DONE FORCE`; `GRIP CLOSE 60` → `ERR silicone load cell disabled`, no motion, no `DONE`.
7. One-shot: after checklist item 4 completes, `GRIP OPEN` then a bare `GRIP CLOSE 90` is position mode again (full speed, stops at 90 mm gap).
8. Clear: `FORCE 1.0`, `FORCE 0`, `GRIP CLOSE 90` → position mode (target was cleared).

- [ ] **Step 4: Record results**

Note any deviation (especially creep-speed feel and stop overshoot — `FORCE_SEEK_SPEED_STEPS_PER_SEC` and `LOADCELL_SAMPLES_IN_USE` are the tuning knobs) in the PR/branch notes before merging.
