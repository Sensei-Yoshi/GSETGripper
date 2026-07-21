// Force-controlled gripper firmware for ground-truth minimum-force collection.
//
// Serial protocol (9600 baud, newline-terminated) — see platformio.ini.
// This extends the LED command-loop pattern already used in the repo
// (Gripper-Arduino/src/main.cpp) with a closed-loop normal-force controller and
// an HX711 load cell behind the STATIONARY finger (our force convention).
//
// Hardware-specific values (pins, LOADCELL_SCALE, control gains, actuator type)
// MUST be calibrated on-site; they are marked with TODO.

#include <Arduino.h>
#include <HX711.h>

// ---- Pin map (TODO: set for your rig) --------------------------------------
const int LOADCELL_DOUT_PIN = 2;
const int LOADCELL_SCK_PIN  = 3;
const int MOTOR_PWM_PIN     = 9;   // actuator driving the moving finger
const int MOTOR_DIR_PIN     = 8;

// ---- Calibration (TODO: from load-cell calibration) ------------------------
const float LOADCELL_SCALE  = 1.0f;   // raw counts per newton
const float FORCE_TOLERANCE = 0.05f;  // N; closed-loop deadband
const float CONTACT_FORCE_N = 0.10f;  // "initial contact" threshold
const unsigned long HOLD_MS = 3000;   // required successful-lift hold time

const unsigned long BAUD_RATE = 9600;

HX711 loadCell;
String command;
float targetForceN = 0.0f;

float readForceN() {
  // Median-ish smoothing; HX711 is ~10-80 Hz.
  return loadCell.get_units(3) / LOADCELL_SCALE;
}

void driveFinger(int pwm, bool closing) {
  digitalWrite(MOTOR_DIR_PIN, closing ? HIGH : LOW);
  analogWrite(MOTOR_PWM_PIN, constrain(pwm, 0, 255));
}

void openGripper() {
  driveFinger(200, /*closing=*/false);
  delay(400);
  driveFinger(0, false);
  targetForceN = 0.0f;
}

void closeUntilContact() {
  unsigned long start = millis();
  while (readForceN() < CONTACT_FORCE_N && millis() - start < 4000) {
    driveFinger(120, /*closing=*/true);
  }
  driveFinger(0, true);
}

// Simple proportional force controller toward targetForceN.
void regulateForce() {
  float err = targetForceN - readForceN();
  if (fabs(err) <= FORCE_TOLERANCE) {
    driveFinger(0, err >= 0);
    return;
  }
  int pwm = (int)constrain(fabs(err) * 200.0f, 40, 255);  // TODO: tune gain
  driveFinger(pwm, /*closing=*/(err > 0));
}

// Standardized lift: hold the target force and check the object does not slip.
bool attemptLift() {
  unsigned long start = millis();
  // TODO: command the arm/stage to lift here; this loop maintains grip force.
  while (millis() - start < HOLD_MS) {
    regulateForce();
    if (readForceN() < CONTACT_FORCE_N) {
      return false;  // lost contact -> dropped
    }
  }
  return true;
}

void processCommand(const String& msg) {
  if (msg.startsWith("SET_FORCE")) {
    targetForceN = msg.substring(9).toFloat();
    regulateForce();
    Serial.println("OK");
  } else if (msg == "CLOSE") {
    closeUntilContact();
    Serial.println("OK");
  } else if (msg == "OPEN") {
    openGripper();
    Serial.println("OK");
  } else if (msg == "READ") {
    Serial.println(readForceN(), 3);
  } else if (msg == "LIFT") {
    Serial.println(attemptLift() ? "HELD" : "DROPPED");
  } else {
    Serial.println("ERR");
  }
}

void setup() {
  pinMode(MOTOR_PWM_PIN, OUTPUT);
  pinMode(MOTOR_DIR_PIN, OUTPUT);
  loadCell.begin(LOADCELL_DOUT_PIN, LOADCELL_SCK_PIN);
  loadCell.tare();
  Serial.begin(BAUD_RATE);
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      command.trim();
      processCommand(command);
      command = "";
    } else {
      command += c;
    }
  }
}
