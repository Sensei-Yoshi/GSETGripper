#include <AccelStepper.h>

const unsigned long BAUD_RATE = 9600;

// ---- Z axis (lead screw) ----
const int Z_A_STEP_PIN = 2;
const int Z_A_DIR_PIN = 3;
AccelStepper stepperZA(AccelStepper::DRIVER, Z_A_STEP_PIN, Z_A_DIR_PIN);

const int Z_B_STEP_PIN = 4;
const int Z_B_DIR_PIN = 5;
AccelStepper stepperZB(AccelStepper::DRIVER, Z_B_STEP_PIN, Z_B_DIR_PIN);

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

  stepperZB.setMaxSpeed(Z_MAX_SPEED_STEPS_PER_SEC);
  stepperZB.setAcceleration(Z_ACCELERATION_STEPS_PER_SEC2);
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

  if (zMovePending && stepperZA.distanceToGo() == 0 && stepperZB.distanceToGo() == 0) {
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
  stepperZB.moveTo(targetSteps);
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
