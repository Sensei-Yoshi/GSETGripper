#include <AccelStepper.h>

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
//
// Z 0 is wherever the arm physically is when the board powers on/resets --
// there's no homing routine or limit switch yet, so it is not tied to any
// fixed real-world height until one is added. For reference: the motor is
// mounted 127 mm (5 in) above the ground, so the nut's reachable band sits
// roughly 127-527 mm above ground.
const float Z_STEPS_PER_MM = 200.0;
const long Z_MIN_STEPS = 0;
// TODO: measure actual usable travel. The screw is 400 mm long, but the nut
// block's own length plus motor-end clearance eat into that -- expect 355-375
// mm. 350 mm is a conservative placeholder until measured.
const long Z_MAX_STEPS = 70000; // 350 mm * 200 steps/mm

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
// The two gripper heads are mounted 80 deg apart:
//   1600 * (80 / 360) = 355.6 steps -> rounded to 356 (+0.1 deg). Moves are
//   absolute, so that rounding offset is fixed and never accumulates.
// TODO: confirm which head sits at 0 deg -- swapping these two constants is
// the whole fix if gecko and silicone are the other way round.
const long SELECT_GEKKO_STEPS = 0;      // head A, 0 deg
const long SELECT_SILICONE_STEPS = 430; // head B, 
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
// GRIP_OPEN_STEPS is the fully-open home position. GRIP_MAX_OPENING_MM is the
// object width (mm) that corresponds to fully open jaws; GRIP_STEPS_PER_MM
// converts the remaining jaw travel needed to close around a narrower object
// into steps.
const long GRIP_OPEN_STEPS = 0;
const float GRIP_MAX_OPENING_MM = 60.0; // TODO: measure the fully-open jaw gap
const float GRIP_STEPS_PER_MM = 200.0;
const long GRIP_MIN_STEPS = 0;
// Fully-closed travel = GRIP_MAX_OPENING_MM * GRIP_STEPS_PER_MM. Update this
// alongside GRIP_MAX_OPENING_MM when the jaw gap is measured.
const long GRIP_MAX_STEPS = 12000;

bool gripMovePending = false;

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
  stepperZA.setPinsInverted(true, false, false);
  stepperZB.setPinsInverted(true, false, false);  
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
    float targetHeightMM;
    if (!parseFloatStrict(arg, targetHeightMM)) {
      sendErr("bad value");
      return;
    }
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

void moveZTo(float targetHeightMM) {
  long targetSteps = clampSteps((long)(targetHeightMM * Z_STEPS_PER_MM), Z_MIN_STEPS, Z_MAX_STEPS, "Z");
  stepperZA.moveTo(targetSteps);
  stepperZB.moveTo(targetSteps);
  zMovePending = true;
}

void moveSelectTo(long targetSteps) {
  long clamped = clampSteps(targetSteps, SELECT_MIN_STEPS, SELECT_MAX_STEPS, "SELECT");
  stepperSelect.moveTo(clamped);
  selectMovePending = true;
}

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
