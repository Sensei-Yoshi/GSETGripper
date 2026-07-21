#include <AccelStepper.h>

const unsigned long BAUD_RATE = 9600;

//set tostep/dir pins and wiring type.
const int STEP_PIN = 2;
const int DIR_PIN = 3;
AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

//tune for motor/rig.
const float MAX_SPEED_STEPS_PER_SEC = 800.0;
const float ACCELERATION_STEPS_PER_SEC2 = 400.0;

// TODO: calibrate for the real gantry (lead screw pitch, microstepping, etc.).
// Z 0 is wherever the arm physically is when the board powers on/resets --
// there's no homing routine or limit switch yet, so it is not tied to any
// fixed real-world height until one is added.
const float STEPS_PER_MM = 10.0;

String command;

void setup() {
  Serial.begin(BAUD_RATE);

  stepper.setMaxSpeed(MAX_SPEED_STEPS_PER_SEC);
  stepper.setAcceleration(ACCELERATION_STEPS_PER_SEC2);
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
  stepper.run();
}

void processCommand(const String& message) {
  if (message.startsWith("MOVE ")) {
    long targetSteps = message.substring(5).toInt();
    stepper.moveTo(targetSteps);
  } else if (message.startsWith("Z ")) {
    float targetHeightMM = message.substring(2).toFloat();
    long targetSteps = (long)(targetHeightMM * STEPS_PER_MM);
    stepper.moveTo(targetSteps);
    Serial.print("Moving to Z=");
    Serial.print(targetHeightMM);
    Serial.print("mm (steps=");
    Serial.print(targetSteps);
    Serial.println(")");
  }
  // TODO: add other commands as needed (e.g. HOME, STOP, SPEED <n>).
}
