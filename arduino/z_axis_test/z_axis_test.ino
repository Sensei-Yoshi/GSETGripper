// Standalone Z-axis test sketch.
// Takes an object height over serial ("Z <height_mm>") and drives both Z
// steppers (ZA, ZB) DOWN from the top of travel so the gripper ends up
// gripping 1 inch below the top of the object. No SELECT/GRIP axes -- Z only.
//
// Wire protocol matches arduino/main/main.ino exactly, so the existing
// camera/depth_closest_read.py can talk to this sketch unmodified.
//
// STARTING POSITION (required): before powering on, the carriage must be
// parked at the TOP of its travel. There is no homing switch, so step 0 is
// simply wherever the carriage sits at reset -- all the geometry below
// assumes that is the top.

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

const float Z_MAX_SPEED_STEPS_PER_SEC = 3000.0;      // 12.5 mm/s
const float Z_ACCELERATION_STEPS_PER_SEC2 = 9000.0;  // 25 mm/s^2

// Calibrated for: NEMA 17, 1.8 deg (200 full steps/rev), integrated Tr8x8 (P2)
// lead screw (4-start, 8 mm lead), TMC2209 driver in standalone mode at 1/8
// microstepping (MS1/MS2 both low).
//   (200 steps/rev * 8 microsteps) / 8 mm per rev = 200 steps/mm
const float Z_STEPS_PER_MM = 200.0;

// If the carriage moves UP when it should move down, flip this to -1
// (or swap one motor coil pair). Both motors follow the same sign.
const int Z_DOWN_SIGN = -1;

// ---- Rig geometry (all heights in mm above the ground) ----
const float MOTOR_MOUNT_HEIGHT_MM = 200.0;       
const float LEAD_SCREW_LENGTH_MM = 305.0;        
// Carriage top at the start (top of travel) = 127 + 400 = 527 mm.
const float CARRIAGE_TOP_START_MM = MOTOR_MOUNT_HEIGHT_MM + LEAD_SCREW_LENGTH_MM;
const float GRIPPER_BELOW_CARRIAGE_TOP_MM = 152.4;  // 6 in: gripping point below carriage top
// Gripper height at the start = 527 - 152.4 = 374.6 mm above ground.
const float GRIPPER_START_HEIGHT_MM = CARRIAGE_TOP_START_MM - GRIPPER_BELOW_CARRIAGE_TOP_MM;
const float GRIP_BELOW_OBJECT_TOP_MM = 25.4;     // 1 in: grip below the object's top

// Mechanical travel: the carriage descends the full screw, from carriage top
// at 527 mm down to the motor mount at 127 mm -- 400 mm. Trim this if the
// nut block or coupling turns out to eat into it on the real rig.
const float Z_USABLE_TRAVEL_MM = 305.0;
// Ground limit: never drive the gripping point below the ground. From the
// 374.6 mm start height that allows at most 374.6 mm of descent, which the
// gripper reaches 25.4 mm BEFORE the carriage hits its mechanical stop --
// so this is the limit that actually binds.
const float Z_TRAVEL_TO_GROUND_MM = GRIPPER_START_HEIGHT_MM;
const long Z_MIN_STEPS = 0;  // top of travel (start)
const long Z_MAX_STEPS = (long)(min(Z_USABLE_TRAVEL_MM, Z_TRAVEL_TO_GROUND_MM) * Z_STEPS_PER_MM);

bool zMovePending = false;
String command;

void setup() {
  Serial.begin(BAUD_RATE);

  pinMode(Z_A_EN_PIN, OUTPUT); 
  digitalWrite(Z_A_EN_PIN, LOW);
  pinMode(Z_B_EN_PIN, OUTPUT); 
  digitalWrite(Z_B_EN_PIN, LOW);

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
    float objectHeightMM;
    if (!parseFloatStrict(arg, objectHeightMM)) {
      sendErr("bad value");
      return;
    }
    moveToObjectHeight(objectHeightMM);
  } else {
    sendErr("unknown command");
  }
}

// Convert an object height (mm above ground) into downward carriage travel.
//   target gripper height = object height - 25.4  (grip 1 in below the top)
//   down travel = gripper start height - target gripper height
// With this rig's numbers that reduces to: down travel = 400 - object height.
void moveToObjectHeight(float objectHeightMM) {
  float targetGripperHeightMM = objectHeightMM - GRIP_BELOW_OBJECT_TOP_MM;
  float downTravelMM = GRIPPER_START_HEIGHT_MM - targetGripperHeightMM;

  long targetSteps = clampSteps((long)(downTravelMM * Z_STEPS_PER_MM), Z_MIN_STEPS, Z_MAX_STEPS, "Z");
  stepperZA.moveTo(Z_DOWN_SIGN * targetSteps);
  stepperZB.moveTo(Z_DOWN_SIGN * targetSteps);
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
