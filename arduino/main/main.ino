#include <AccelStepper.h>
#include <HX711_ADC.h>
#include <EEPROM.h>

const unsigned long BAUD_RATE = 9600;

// ---- ZA to E0 ----
const int Z_A_STEP_PIN = 26;
const int Z_A_DIR_PIN  = 28;
const int Z_A_EN_PIN   = 24;   // A8
AccelStepper stepperZA(AccelStepper::DRIVER, Z_A_STEP_PIN, Z_A_DIR_PIN);

// ZB -> E1 driver socket
const int Z_B_STEP_PIN = 36;
const int Z_B_DIR_PIN  = 34;
const int Z_B_EN_PIN   = 30;
AccelStepper stepperZB(AccelStepper::DRIVER, Z_B_STEP_PIN, Z_B_DIR_PIN);

const float Z_MAX_SPEED_STEPS_PER_SEC = 2500.0;      // 12.5 mm/s
const float Z_ACCELERATION_STEPS_PER_SEC2 = 5000.0;  // 25 mm/s^2; reaches full speed in 0.5 s over 3.1 mm

// Calibrated for: NEMA 17, 1.8 deg (200 full steps/rev), integrated Tr8x8 (P2)
// lead screw (4-start, 8 mm lead), TMC2209 driver in standalone mode at 1/8
// microstepping (MS1/MS2 both low).
//   (200 steps/rev * 8 microsteps) / 8 mm per rev = 200 steps/mm
const float Z_STEPS_PER_MM = 200.0;
const int Z_DOWN_SIGN = -1;

// ---- Z geometry, measured 2026-07-24 (all imperial values * 25.4) ----
//
//   floor ---------------------------------------------- 0 mm
//     |  8.20 in = 208.28 mm   bottom of usable screw travel
//     |  screw travel 12.00 in = 304.80 mm
//   carriage top of travel ---------------------------- 513.08 mm (20.20 in)
//   gripper hangs 10.25 in = 260.35 mm below the cross bar, so with the
//   carriage at the top the gripper bottom sits at 252.73 mm (9.95 in).
//
// The gripper bottom would reach the floor with the carriage still 52.07 mm
// above the bottom of the screw (208.28 - 260.35 = -52.07), so the bottom
// 52 mm of screw travel is mechanically unusable -- driving into it puts the
// gripper through the floor. Z_MAX_STEPS below is the hard cap that prevents
// that; there is no limit switch to catch it in hardware.
const float Z_SCREW_TRAVEL_MM = 304.8;               // 12.00 in usable screw length
const float Z_SCREW_BOTTOM_ABOVE_FLOOR_MM = 208.28;  //  8.20 in
const float Z_GRIPPER_BELOW_CARRIAGE_MM = 260.35;    // 10.25 in cross bar to gripper bottom
const float Z_CARRIAGE_MAX_ABOVE_FLOOR_MM =
    Z_SCREW_BOTTOM_ABOVE_FLOOR_MM + Z_SCREW_TRAVEL_MM;  // 513.08 mm (20.20 in)

// Extra stand-off kept between the gripper bottom and the floor. The gripper
// never descends below this, so it doubles as the parking height for objects
// too short to grasp at full suction depth.
const float Z_FLOOR_CLEARANCE_MM = 10.0;

// Reachable band for the GRIPPER BOTTOM above the floor. Note this is not what
// "Z <mm>" carries -- see the grasp placement block below.
const float Z_MAX_GRIPPER_HEIGHT_MM =
    Z_CARRIAGE_MAX_ABOVE_FLOOR_MM - Z_GRIPPER_BELOW_CARRIAGE_MM;  // 252.73 mm (9.95 in)
const float Z_MIN_GRIPPER_HEIGHT_MM = Z_FLOOR_CLEARANCE_MM;

// Step 0 is the TOP of travel, not the floor: there is no homing routine or
// limit switch, so the firmware assumes the carriage is parked at the top of
// the screw when the board powers on/resets. Park it there before plugging in
// or opening the serial port (opening the port resets the board). Steps count
// downward from there, so a bigger step count = a lower gripper.
const long Z_MIN_STEPS = 0;
const long Z_MAX_STEPS =
    (long)((Z_MAX_GRIPPER_HEIGHT_MM - Z_MIN_GRIPPER_HEIGHT_MM) * Z_STEPS_PER_MM);  // 48545

// ---- Grasp placement ----
// "Z <mm>" carries the height of the TOP OF THE OBJECT above the floor -- that
// is what the depth camera measures and sends. The firmware, not the camera
// script, turns that into a carriage position.
//
// The suction strip runs 4.25 in up from the bottom of the gripper, so the
// gripper bottom is placed that far below the top of the object to put the
// whole strip in contact. Objects shorter than this cannot fit the full strip,
// so they are grasped from Z_MIN_GRIPPER_HEIGHT_MM instead -- as low as the
// gripper is allowed to go, which maximises the contact that does fit.
const float GRASP_DEPTH_BELOW_TOP_MM = 107.95;  // 4.25 in suction strip length

bool zMovePending = false;

// ---- SELECT axis -> X driver socket ----
const int SELECT_STEP_PIN = 54;   // A0
const int SELECT_DIR_PIN  = 55;   // A1
const int SELECT_EN_PIN   = 38;
AccelStepper stepperSelect(AccelStepper::DRIVER, SELECT_STEP_PIN, SELECT_DIR_PIN);

const float SELECT_MAX_SPEED_STEPS_PER_SEC = 1600.0;  // 1 turret rev/s (60 RPM)
const float SELECT_ACCELERATION_STEPS_PER_SEC2 = 2000.0;

// Rotary turret, direct drive (no gearbox or belt), same 1.8 deg motor and 1/8
// microstepping as Z: 200 steps/rev * 8 = 1600 microsteps per turret rev.
// The two gripper heads are mounted 430 steps apart (measured), which is
//   430 / 1600 * 360 = 96.75 deg.
// Moves are absolute, so no rounding error accumulates across selections.
// TODO: confirm which head sits at 0 deg -- swapping these two constants is
// the whole fix if gecko and silicone are the other way round.
const long SELECT_GEKKO_STEPS = 0;      // head A, 0 deg
const long SELECT_SILICONE_STEPS = 430; // head B, 96.75 deg
const long SELECT_MIN_STEPS = 0;
const long SELECT_MAX_STEPS = 430;

bool selectMovePending = false;

// ---- GRIP axis -> Y driver socket ----
const int GRIP_STEP_PIN = 60;   // A6
const int GRIP_DIR_PIN  = 61;   // A7
const int GRIP_EN_PIN   = 56;   // A2
AccelStepper stepperGrip(AccelStepper::DRIVER, GRIP_STEP_PIN, GRIP_DIR_PIN);

const float GRIP_MAX_SPEED_STEPS_PER_SEC = 2500.0;      // 12.5 mm/s of jaw travel
const float GRIP_ACCELERATION_STEPS_PER_SEC2 = 5000.0;  // 25 mm/s^2

// Separate Tr8x8 (P2) lead screw with the same spec as Z (4-start, 8 mm lead),
// same 1.8 deg motor, same 1/8 microstepping, coupled 1:1 with no reduction:
//   (200 steps/rev * 8 microsteps) / 8 mm per rev = 200 steps/mm
// Only ONE jaw moves (the opposing jaw is rigidly fixed to the frame), so jaw
// opening changes 1:1 with nut travel -- hence no factor of 2 below or in
// gripStepsForWidth().
//
// Measured 2026-07-24: the grip screw has 4.00 in = 101.6 mm of usable travel,
// and running it all the way from the open end closes the jaws to a zero gap.
// The jaw gap therefore equals the travel REMAINING, so a fully-open gap of
// 101.6 mm and a closing travel of 101.6 mm are the same measurement:
//   gap = GRIP_MAX_OPENING_MM - (steps / GRIP_STEPS_PER_MM)
//
// Step 0 is the fully-open end, which -- like Z -- is assumed, not homed. The
// jaws must be physically fully open whenever the board powers on or resets,
// and opening the serial port resets the board. "GRIP OPEN" returns to that
// flashed-at position.
const long GRIP_OPEN_STEPS = 0;
const float GRIP_MAX_OPENING_MM = 101.6; // 4.00 in fully-open jaw gap
const float GRIP_STEPS_PER_MM = 200.0;
// FLIP THIS if the jaws open when they should close. +1 or -1, nothing else.
// Test with a small move (GRIP CLOSE 90) before trusting a full-travel one.
// The equivalent knob is Z_DOWN_SIGN for the Z axis and the setPinsInverted()
// call in setup() for SELECT.
const int GRIP_CLOSE_SIGN = 1;
const long GRIP_MIN_STEPS = 0;
// Fully closed: the whole 101.6 mm of travel consumed.
const long GRIP_MAX_STEPS = (long)(GRIP_MAX_OPENING_MM * GRIP_STEPS_PER_MM);  // 20320

bool gripMovePending = false;

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

String command;

void setup() {
  Serial.begin(BAUD_RATE);

  pinMode(Z_A_EN_PIN, OUTPUT); 
  digitalWrite(Z_A_EN_PIN, LOW);
  pinMode(Z_B_EN_PIN, OUTPUT); 
  digitalWrite(Z_B_EN_PIN, LOW);
  pinMode(SELECT_EN_PIN, OUTPUT);
  digitalWrite(SELECT_EN_PIN, LOW);
  pinMode(GRIP_EN_PIN, OUTPUT);
  digitalWrite(GRIP_EN_PIN, LOW);

  stepperZA.setMaxSpeed(Z_MAX_SPEED_STEPS_PER_SEC);
  stepperZA.setAcceleration(Z_ACCELERATION_STEPS_PER_SEC2);

  stepperZB.setMaxSpeed(Z_MAX_SPEED_STEPS_PER_SEC);
  stepperZB.setAcceleration(Z_ACCELERATION_STEPS_PER_SEC2);

  stepperSelect.setMaxSpeed(SELECT_MAX_SPEED_STEPS_PER_SEC);
  stepperSelect.setAcceleration(SELECT_ACCELERATION_STEPS_PER_SEC2);

  stepperGrip.setMaxSpeed(GRIP_MAX_SPEED_STEPS_PER_SEC);
  stepperGrip.setAcceleration(GRIP_ACCELERATION_STEPS_PER_SEC2);

  stepperSelect.setPinsInverted(true, false, false);

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
  stepperZB.run();
  stepperSelect.run();
  stepperGrip.run();

  serviceLoadCells();

  if (zMovePending && stepperZA.distanceToGo() == 0 && stepperZB.distanceToGo() == 0) {
    zMovePending = false;
    Serial.println("DONE Z");
  }
  if (selectMovePending && stepperSelect.distanceToGo() == 0) {
    selectMovePending = false;
    Serial.println("DONE SELECT");
  }
  if (gripMovePending && stepperGrip.distanceToGo() == 0) {
    gripMovePending = false;
    Serial.println("DONE GRIP");
  }
}

void processCommand(const String& message) {
  if (message.length() == 0) {
    return;
  }

  if (message.startsWith("Z ")) {
    String arg = message.substring(2);
    arg.trim();
    float objectTopHeightMM;
    if (!parseFloatStrict(arg, objectTopHeightMM)) {
      sendErr("bad value");
      return;
    }
    moveZForGrasp(objectTopHeightMM);
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
      // A negative width can only come from a bad measurement. Refuse rather
      // than let it drive the jaws past closed onto whatever is between them.
      if (widthMM < 0.0) {
        sendErr("negative width");
        return;
      }
      moveGripToSteps(gripStepsForWidth(widthMM));
    } else {
      sendErr("unknown grip command");
    }
  }
  else if (message.startsWith("FORCE")){
    String arg = message.substring(6);
    arg.trim();
  }
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
  else {
    sendErr("unknown command");
  }
}

// Internal axis move -- reached through moveZForGrasp, not directly from the
// serial protocol. targetHeightMM is the desired height of the gripper bottom
// above the floor. It is clamped in millimetres first so the WARN reports a
// real-world height, then converted to a downward step count from the
// top-of-travel origin.


void moveZTo(float targetHeightMM) {
  float clampedHeightMM = targetHeightMM;
  if (clampedHeightMM < Z_MIN_GRIPPER_HEIGHT_MM) {
    clampedHeightMM = Z_MIN_GRIPPER_HEIGHT_MM;
  } else if (clampedHeightMM > Z_MAX_GRIPPER_HEIGHT_MM) {
    clampedHeightMM = Z_MAX_GRIPPER_HEIGHT_MM;
  }
  if (clampedHeightMM != targetHeightMM) {
    Serial.print("WARN Z clamped to ");
    Serial.print(clampedHeightMM, 1);
    Serial.println(" mm above floor");
  }

  float descentMM = Z_MAX_GRIPPER_HEIGHT_MM - clampedHeightMM;
  // clampSteps is a redundant backstop -- the mm clamp above already keeps this
  // in range -- but it catches any future arithmetic slip before the motor moves.
  long targetSteps = clampSteps((long)(descentMM * Z_STEPS_PER_MM), Z_MIN_STEPS, Z_MAX_STEPS, "Z");
  stepperZA.moveTo(Z_DOWN_SIGN * targetSteps);
  stepperZB.moveTo(Z_DOWN_SIGN * targetSteps);
  zMovePending = true;
}

// Handles "Z <mm>". objectTopHeightMM is the height of the TOP of the object
// above the floor -- what the camera sends. Places the gripper bottom one suction-strip
// length below that so the whole strip contacts the object; if the object is
// too short for that to clear the floor, grasps from the lowest allowed height
// instead. Both branches still pass through moveZTo's clamp.
void moveZForGrasp(float objectTopHeightMM) {
  float targetHeightMM = objectTopHeightMM - GRASP_DEPTH_BELOW_TOP_MM;
  if (targetHeightMM < Z_MIN_GRIPPER_HEIGHT_MM) {
    targetHeightMM = Z_MIN_GRIPPER_HEIGHT_MM;
  }
  moveZTo(targetHeightMM);
}

void moveSelectTo(long targetSteps) {
  long clamped = clampSteps(targetSteps, SELECT_MIN_STEPS, SELECT_MAX_STEPS, "SELECT");
  stepperSelect.moveTo(clamped);
  selectMovePending = true;
}

// targetSteps counts travel away from the fully-open end, so it is always >= 0.
// GRIP_CLOSE_SIGN decides which way the motor turns to consume that travel.
void moveGripToSteps(long targetSteps) {
  long clamped = clampSteps(targetSteps, GRIP_MIN_STEPS, GRIP_MAX_STEPS, "GRIP");
  stepperGrip.moveTo(GRIP_CLOSE_SIGN * clamped);
  gripMovePending = true;
}

// Converts an object width into a jaw position: close by however much travel
// is left over once the object's width is accounted for, leaving a gap equal
// to the width. An object wider than the jaws open (travel would go negative)
// leaves them fully open rather than reaching for an impossible position.
long gripStepsForWidth(float widthMM) {
  float travelMM = GRIP_MAX_OPENING_MM - widthMM;
  if (travelMM < 0) {
    travelMM = 0;
  }
  return GRIP_OPEN_STEPS + (long)(travelMM * GRIP_STEPS_PER_MM);
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
