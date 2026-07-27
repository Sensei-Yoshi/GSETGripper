// Lowest point: 180
//
// BUILD 6 -- adds JOGLIVE, a browser-driven live jog that starts on keydown
// and stops on keyup, with a serial watchdog so a dropped STOP cannot leave an
// axis running. Works alongside the existing keyboard JOG mode and all line
// commands. See gantry-jog-panel.html for the matching Web Serial UI.
//
// NOTE: soft travel limits removed from BOTH Z and GRIP. The only remaining
// software guard is GRIP_FORCE_LIMIT_G, and it acts on tightening only. Every
// other direction runs until it hits a mechanical stop and stalls. No endstops.

#include <Arduino.h>
#include <AccelStepper.h>
#include <HX711_ADC.h>
#include <EEPROM.h>

// 9600 is too slow to stream two channels at 20 Hz -- each line is ~80 bytes,
// and 9600 baud only carries ~960 B/s. Set the Serial Monitor to 115200.
const unsigned long BAUD_RATE = 115200;

// ---- ZA -> E0 driver socket ----
const int Z_A_STEP_PIN = 26;
const int Z_A_DIR_PIN = 28;
const int Z_A_EN_PIN = 24;
AccelStepper stepperZA(AccelStepper::DRIVER, Z_A_STEP_PIN, Z_A_DIR_PIN);

// ---- ZB -> E1 driver socket ----
const int Z_B_STEP_PIN = 36;
const int Z_B_DIR_PIN = 34;
const int Z_B_EN_PIN = 30;
AccelStepper stepperZB(AccelStepper::DRIVER, Z_B_STEP_PIN, Z_B_DIR_PIN);

// AccelStepper's practical ceiling on a 16 MHz AVR is roughly 4000 steps/s.
// Asking for more does not fail loudly, it just quietly runs slower than
// commanded. At 200 steps/mm, 4000 steps/s = 20 mm/s.
const float Z_MAX_SPEED_STEPS_PER_SEC = 4000.0;

// Aggressive but not instant: 4000/30000 = 0.13 s to full speed, covering
// 0.27 mm. Feels immediate without asking the rotor to teleport. If a move
// starts with an audible buzz instead of a clean ramp, this is too high.
const float Z_ACCELERATION_STEPS_PER_SEC2 = 30000.0;

// NEMA 17 1.8 deg (200 full steps/rev), Tr8x8 (P2) lead screw (4-start,
// 8 mm lead), TMC2209 standalone at 1/8 microstepping (MS1/MS2 low):
// (200 steps/rev * 8 microsteps) / 8 mm per rev = 200 steps/mm
const float Z_STEPS_PER_MM = 200.0;

// Direction. -1 makes a POSITIVE commanded value move the carriage DOWN.
// If Z 150 drives the carriage UP, flip this to +1. Do NOT also use
// setPinsInverted() on the Z steppers -- two flips cancel out.
const int Z_DOWN_SIGN = -1;

// Travel limits removed. Constants kept for reference/other math only.
const long Z_MIN_STEPS = 0;
const long Z_MAX_STEPS = 70000; // 350 mm * 200 steps/mm
const float Z_MAX_MM = 350.0;

bool zMovePending = false;

// ---- SELECT axis -> X driver socket ----
const int SELECT_STEP_PIN = 54; // A0
const int SELECT_DIR_PIN = 55; // A1
const int SELECT_EN_PIN = 38;
AccelStepper stepperSelect(AccelStepper::DRIVER, SELECT_STEP_PIN, SELECT_DIR_PIN);

const float SELECT_MAX_SPEED_STEPS_PER_SEC = 1600.0; // 1 turret rev/s (60 RPM)
const float SELECT_ACCELERATION_STEPS_PER_SEC2 = 2000.0;

// Rotary turret, direct drive (no gearbox or belt), same 1.8 deg motor and 1/8
// microstepping as Z: 200 steps/rev * 8 = 1600 microsteps per turret rev.
const long SELECT_GEKKO_STEPS = 0; // head A, 0 deg
const long SELECT_SILICONE_STEPS = 430; // head B
const long SELECT_MIN_STEPS = 0;
const long SELECT_MAX_STEPS = 430;

bool selectMovePending = false;

// ---- GRIP axis -> Y driver socket ----
const int GRIP_STEP_PIN = 60; // A6
const int GRIP_DIR_PIN = 61; // A7
const int GRIP_EN_PIN = 56; // A2
AccelStepper stepperGrip(AccelStepper::DRIVER, GRIP_STEP_PIN, GRIP_DIR_PIN);

const float GRIP_MAX_SPEED_STEPS_PER_SEC = 2500.0; // 12.5 mm/s of jaw travel
const float GRIP_ACCELERATION_STEPS_PER_SEC2 = 20000.0;

// Separate Tr8x8 (P2) lead screw, same spec as Z, coupled 1:1 with no
// reduction: (200 steps/rev * 8 microsteps) / 8 mm per rev = 200 steps/mm
// Only ONE jaw moves, so jaw opening changes 1:1 with nut travel.
const long GRIP_OPEN_STEPS = 0;
const float GRIP_MAX_OPENING_MM = 60.0; // TODO: measure the fully-open jaw gap
const float GRIP_STEPS_PER_MM = 200.0;
// Travel limits removed. Constants kept for reference/other math only.
const long GRIP_MIN_STEPS = 0;
const long GRIP_MAX_STEPS = 16000;

// Sign convention for relative jogs ("GRIP +10" / "GRIP -5"):
// +1 means a POSITIVE value TIGHTENS (jaw gap shrinks by that many mm).
// Flip to -1 if you'd rather have + mean "open wider".
const int GRIP_TIGHTEN_SIGN = +1;

bool gripMovePending = false;

// =================== JOG MODE ===================
// Type JOG to enter, q to leave. Inside jog mode single keystrokes move axes
// and no line commands are parsed -- that avoids the collision where the 'S'
// in "SELECT SILICONE" would otherwise fire a jog.
//
// A single keypress = one discrete step. Holding a key makes the OS auto-repeat
// send a stream of characters; the second one within JOG_HOLD_DETECT_MS
// switches to continuous motion, and each further character refreshes a
// dead-man timer. Release the key, the stream stops, the axis stops. Nothing
// keeps moving if the terminal crashes or the window loses focus.
//
// The Arduino IDE Serial Monitor cannot hold-repeat (it only sends on Enter),
// so from the IDE you get tap-stepping only. For continuous jog use PuTTY, or
// screen /dev/cu.usbmodemXXXX 115200 on macOS/Linux -- or use the browser panel.
const float Z_JOG_STEP_FINE_MM = 0.25;
const float Z_JOG_STEP_FAST_MM = 2.0;
const float Z_JOG_SPEED_FINE = 400.0; // 2 mm/s
const float Z_JOG_SPEED_FAST = 4000.0; // 20 mm/s

const float GRIP_JOG_STEP_FINE_MM = 0.10;
const float GRIP_JOG_STEP_FAST_MM = 1.0;
const float GRIP_JOG_SPEED_FINE = 200.0; // 1 mm/s
const float GRIP_JOG_SPEED_FAST = 2000.0; // 10 mm/s

// Second keypress within this window means "held", not "tapped twice". Raise it
// if your OS auto-repeat delay is long and holding a key only ever single-steps.
const unsigned long JOG_HOLD_DETECT_MS = 700;


// No refreshing character within this window while continuous -> stop.
// Auto-repeat is typically 25-30 chars/s (~35 ms apart), so 150 ms is safe.
const unsigned long JOG_TIMEOUT_MS = 150;

// How far ahead of the current position to park the target during continuous
// jog. Long enough that the axis reaches cruise, short enough that a missed
// refresh costs little coasting.
const float JOG_LOOKAHEAD_S = 0.30;

// Grip tightening stops here. This lives in firmware because the round trip
// through a terminal is 100 ms+ and the HX711 average lags further -- by the
// time a human reacts the jaw has already loaded up. Set well below anything
// that could damage a finger or the load cell.
const float GRIP_FORCE_LIMIT_G = 1500.0;

struct JogAxis {
 int dir; // -1, 0, +1
 bool continuous;
 bool fast;
 unsigned long lastKeyMS;
 bool speedOverridden;
};
JogAxis jogZ = {0, false, false, 0, false};
JogAxis jogG = {0, false, false, 0, false};

bool jogMode = false;
bool jogFast = false;

// Arrow keys arrive as escape sequences: ESC [ A or, in application cursor
// mode, ESC O A. 0 = idle, 1 = saw ESC, 2 = saw the bracket or O.
byte escState = 0;

// =================== LIVE JOG (browser press/hold) ===================
// Unlike JOG mode this needs no mode switch and no auto-repeat heuristic: the
// panel sends an explicit "go" on keydown and "stop" on keyup, so the firmware
// keeps refreshing a lookahead target until told otherwise. A watchdog guards
// the one failure this exposes -- a dropped STOP. The panel pings while a key
// is held; if no ping arrives within LIVEJOG_TIMEOUT_MS the axis stops itself.
int liveJogZDir = 0; // -1 up, +1 down, 0 idle
int liveJogGripDir = 0; // -1 loosen, +1 tighten, 0 idle
unsigned long liveJogZLastMS = 0;
unsigned long liveJogGripLastMS = 0;
const unsigned long LIVEJOG_TIMEOUT_MS = 400;

// Live jog uses the FAST jog speeds. Change here if you want it slower.
const float LIVEJOG_Z_SPEED = Z_JOG_SPEED_FAST;
const float LIVEJOG_GRIP_SPEED = GRIP_JOG_SPEED_FAST;

// =================== HX711 LOAD CELLS (HX711_ADC) ===================
// Cell 1: DOUT -> A5, SCK -> A9
// Cell 2: DOUT -> A10, SCK -> A11
// A9-A12 are unused by RAMPS (A13-A15 are the thermistors), and AUX-2 has 5V
// and GND on the same header. Each amp needs its own clock line.
const int LC1_DOUT = A5;
const int LC1_SCK = A9;
const int LC2_DOUT = A10;
const int LC2_SCK = A11;

HX711_ADC loadCell1(LC1_DOUT, LC1_SCK);
HX711_ADC loadCell2(LC2_DOUT, LC2_SCK);

const int LOADCELL_COUNT = 2;
HX711_ADC* loadCell[LOADCELL_COUNT] = {&loadCell1, &loadCell2};

// Set an entry false for any cell that is not physically wired yet. An unwired
// HX711 leaves DOUT floating; a floating pin reads LOW much of the time, so
// update() believes a conversion is waiting on EVERY loop pass and does a full
// 24-bit shift-out each time. That starves stepper.run() and the axes crawl.
const bool LOADCELL_ENABLED[LOADCELL_COUNT] = {true, false};

// Flip the sign properly rather than fabs()-ing the reading. fabs() rectifies
// the noise around zero into a positive bias, which corrupts small-force
// readings and your tared baseline.
const bool LOADCELL_REVERSED[LOADCELL_COUNT] = {false, false};

const unsigned long LOADCELL_STABILIZING_MS = 2000;

// Extra smoothing on top of the library's own moving average. 1.0 = none.
const float LOADCELL_ALPHA = 0.15;

const float G_TO_NEWTON = 0.00980665f; // 1 gram-force -> N

// getData() is a moving average over SAMPLES (16 by default) with the highest
// and lowest discarded. On a stock 10 SPS breakout that is a ~1.6 s window --
// fine for settled preload, sluggish for anything transient. Drop this to 4 or
// 8 for faster response, or tie the HX711 RATE pin high for 80 SPS.
const int LOADCELL_SAMPLES_IN_USE = 0; // 0 = leave the library default alone

// ---- settle detection (IDLE -> CLIMBING -> settled, per cell) ----
const float PLACEMENT_DERIVATIVE = 2.0; // grams of jump that arms detection
const float STABLE_PEAK_LIMIT = 0.5; // grams of allowed wobble
const int REQUIRED_STABLE_UPDATES = 5;

enum SettleState { SETTLE_OFF, SETTLE_IDLE, SETTLE_CLIMBING };

struct CellRuntime {
 float filtered;
 float previous;
 float baseline;
 bool haveFirst;
 SettleState settle;
 int stableUpdates;
 bool tarePending;
};
CellRuntime cellRT[LOADCELL_COUNT];

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

// Calibration factor per cell, in raw counts per gram. Cell 1 is your measured
// 655.1. Use a NEGATIVE value if a cell reads backwards under load.
const float CAL_FACTOR[LOADCELL_COUNT] = {655.1, 655.1};

// true = always use CAL_FACTOR above and tare fresh on every boot
// false = restore whatever TARE/CAL last saved to EEPROM
const bool USE_HARDCODED_CAL = true;

bool streaming = false;
unsigned long streamIntervalMS = 50; // 20 Hz
unsigned long lastStreamMS = 0;

// ---- loop rate diagnostic ----
// Healthy is tens of thousands of Hz. A few thousand means something in the
// loop is eating time, and no stepper speed setting will fix that.
unsigned long loopCount = 0;
unsigned long lastRateMS = 0;
unsigned int loopHz = 0;

String command;

void setup() {
 Serial.begin(BAUD_RATE);
 delay(100);

 // RAMPS driver enable pins are ACTIVE LOW -- nothing moves without this.
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

 for (int i = 0; i < LOADCELL_COUNT; i++) {
 cellRT[i].filtered = 0.0;
 cellRT[i].previous = 0.0;
 cellRT[i].baseline = 0.0;
 cellRT[i].haveFirst = false;
 cellRT[i].settle = SETTLE_OFF;
 cellRT[i].stableUpdates = 0;
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
 Serial.print(F("ERR cell "));
 Serial.print(i + 1);
 Serial.println(F(" timed out -- check DOUT/SCK wiring and 5V"));
 }
 if (USE_HARDCODED_CAL) {
 cal.calFactor[i] = CAL_FACTOR[i];
 }
 loadCell[i]->setCalFactor(cal.calFactor[i]);
 if (LOADCELL_SAMPLES_IN_USE > 0) {
 loadCell[i]->setSamplesInUse(LOADCELL_SAMPLES_IN_USE);
 }
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

 // Upload marker -- if you don't see this line in the serial monitor after
 // a reset, the new code did not actually get onto the board.
 Serial.println(F("READY BUILD 6 (jog + livejog, limits off)"));
 calPrint();
 Serial.println(F("HELP for commands, JOG for keyboard control"));
}

void loop() {
 loopCount++;
 if (millis() - lastRateMS >= 1000) {
 loopHz = loopCount;
 loopCount = 0;
 lastRateMS = millis();
 }

 while (Serial.available() > 0) {
 char incoming = Serial.read();

 if (jogMode) {
 handleJogChar(incoming);
 } else if (incoming == '\n') {
 command.trim();
 processCommand(command);
 command = "";
 } else if (incoming != '\r') {
 command += incoming;
 }
 }

 // Must be called as often as possible for AccelStepper to step correctly.
 runSteppers();
 serviceJog();
 serviceLiveJog();
 serviceLoadCells();
 runSteppers();

 if (zMovePending && stepperZA.distanceToGo() == 0 && stepperZB.distanceToGo() == 0) {
 zMovePending = false;
 Serial.println(F("DONE Z"));
 }
 if (selectMovePending && stepperSelect.distanceToGo() == 0) {
 selectMovePending = false;
 Serial.println(F("DONE SELECT"));
 }
 if (gripMovePending && stepperGrip.distanceToGo() == 0) {
 gripMovePending = false;
 Serial.print(F("DONE GRIP opening="));
 Serial.print(gripOpeningMM(), 2);
 Serial.println(F(" mm"));
 }

 if (streaming && (millis() - lastStreamMS >= streamIntervalMS)) {
 lastStreamMS = millis();
 printForceLine();
 }
}

void runSteppers() {
 stepperZA.run();
 stepperZB.run();
 stepperSelect.run();
 stepperGrip.run();
}

// ---------------- LIVE JOG ----------------

void startLiveJogZ(int dir) {
 liveJogZDir = dir;
 liveJogZLastMS = millis();
 stepperZA.setMaxSpeed(LIVEJOG_Z_SPEED);
 stepperZB.setMaxSpeed(LIVEJOG_Z_SPEED);
}

void stopLiveJogZ() {
 if (liveJogZDir != 0) {
 stepperZA.stop();
 stepperZB.stop();
 }
 liveJogZDir = 0;
 stepperZA.setMaxSpeed(Z_MAX_SPEED_STEPS_PER_SEC);
 stepperZB.setMaxSpeed(Z_MAX_SPEED_STEPS_PER_SEC);
}

void startLiveJogGrip(int dir) {
 liveJogGripDir = dir;
 liveJogGripLastMS = millis();
 stepperGrip.setMaxSpeed(LIVEJOG_GRIP_SPEED);
}

void stopLiveJogGrip() {
 if (liveJogGripDir != 0) {
 stepperGrip.stop();
 }
 liveJogGripDir = 0;
 stepperGrip.setMaxSpeed(GRIP_MAX_SPEED_STEPS_PER_SEC);
}

void serviceLiveJog() {
 unsigned long now = millis();

 if (liveJogZDir != 0) {
 if (now - liveJogZLastMS > LIVEJOG_TIMEOUT_MS) {
 stopLiveJogZ();
 Serial.println(F("WARN livejog Z watchdog -- no ping, stopped"));
 } else {
 float lookMM = (LIVEJOG_Z_SPEED / Z_STEPS_PER_MM) * JOG_LOOKAHEAD_S;
 float target = zPositionMM() + liveJogZDir * lookMM;
 stepperZA.moveTo(zStepsForMM(target));
 stepperZB.moveTo(zStepsForMM(target));
 }
 }

 if (liveJogGripDir != 0) {
 if (now - liveJogGripLastMS > LIVEJOG_TIMEOUT_MS) {
 stopLiveJogGrip();
 Serial.println(F("WARN livejog GRIP watchdog -- no ping, stopped"));
 } else if (liveJogGripDir > 0 && gripForceExceeded()) {
 stopLiveJogGrip();
 } else {
 float lookMM = (LIVEJOG_GRIP_SPEED / GRIP_STEPS_PER_MM) * JOG_LOOKAHEAD_S;
 long lookSteps = (long)(lookMM * GRIP_STEPS_PER_MM);
 long target = stepperGrip.currentPosition() + GRIP_TIGHTEN_SIGN * liveJogGripDir * lookSteps;
 stepperGrip.moveTo(target);
 }
 }
}

// ---------------- JOG MODE ----------------

void enterJogMode() {
 jogMode = true;
 escState = 0;
 jogFast = false;
 Serial.println(F("JOG MODE ON"));
 printJogHelp();
}

void exitJogMode() {
 stopZJog();
 stopGripJog();
 jogMode = false;
 escState = 0;
 Serial.println(F("JOG MODE OFF"));
}

void printJogHelp() {
 Serial.println(F(" up/down or w/s : Z raise / lower"));
 Serial.println(F(" left/right or a/d: gripper loosen / tighten"));
 Serial.println(F(" tap = one step, hold = continuous"));
 Serial.println(F(" f = toggle fine/fast space = stop p = position q = quit"));
}

void handleJogChar(char c) {
 // Arrow keys: ESC [ A/B/C/D, or ESC O A/B/C/D in application cursor mode.
 if (escState == 1) {
 escState = (c == '[' || c == 'O') ? 2 : 0;
 return;
 }
 if (escState == 2) {
 escState = 0;
 switch (c) {
 case 'A': jogKey(true, -1); return; // up -> Z raises
 case 'B': jogKey(true, +1); return; // down -> Z lowers
 case 'D': jogKey(false, -1); return; // left -> loosen
 case 'C': jogKey(false, +1); return; // right -> tighten
 default: return;
 }
 }
 if (c == 27) { // ESC
 escState = 1;
 return;
 }

 switch (c) {
 case 'w': case 'W': jogKey(true, -1); break;
 case 's': case 'S': jogKey(true, +1); break;
 case 'a': case 'A': jogKey(false, -1); break;
 case 'd': case 'D': jogKey(false, +1); break;
 case 'f': case 'F':
 jogFast = !jogFast;
 Serial.println(jogFast ? F("JOG FAST") : F("JOG FINE"));
 break;
 case ' ':
 stopZJog();
 stopGripJog();
 Serial.println(F("JOG STOP"));
 break;
 case 'p': case 'P':
 printPosition();
 break;
 case '?':
 printJogHelp();
 break;
 case 'q': case 'Q':
 exitJogMode();
 break;
 default:
 break; // ignore newlines and anything unmapped
 }
}

// One keypress. If this is a repeat of the same direction inside the hold
// window, escalate to continuous motion instead of stacking discrete steps.
void jogKey(bool isZ, int dir) {
 JogAxis& j = isZ ? jogZ : jogG;
 unsigned long now = millis();
 bool isRepeat = (j.dir == dir) && (now - j.lastKeyMS <= JOG_HOLD_DETECT_MS);

 j.dir = dir;
 j.fast = jogFast;
 j.lastKeyMS = now;

 if (!isRepeat) {
 j.continuous = false;
 if (isZ) {
 jogZDiscrete(dir);
 } else {
 jogGripDiscrete(dir);
 }
 } else if (!j.continuous) {
 j.continuous = true;
 if (isZ) {
 stepperZA.setMaxSpeed(jogZSpeed());
 stepperZB.setMaxSpeed(jogZSpeed());
 jogZ.speedOverridden = true;
 } else {
 stepperGrip.setMaxSpeed(jogGripSpeed());
 jogG.speedOverridden = true;
 }
 }
 // Already continuous: the call above refreshed lastKeyMS, which is the point.
}

float jogZSpeed() {
 return jogZ.fast ? Z_JOG_SPEED_FAST : Z_JOG_SPEED_FINE;
}

float jogGripSpeed() {
 return jogG.fast ? GRIP_JOG_SPEED_FAST : GRIP_JOG_SPEED_FINE;
}

void serviceJog() {
 unsigned long now = millis();

 if (jogZ.continuous) {
 if (now - jogZ.lastKeyMS > JOG_TIMEOUT_MS) {
 stopZJog();
 } else {
 refreshZJog();
 }
 }
 if (jogG.continuous) {
 if (now - jogG.lastKeyMS > JOG_TIMEOUT_MS) {
 stopGripJog();
 } else {
 refreshGripJog();
 }
 }
}

// Park the target a fixed time ahead of where we are. The target keeps running
// away, so the axis accelerates to cruise and stays there.
void refreshZJog() {
 float lookMM = (jogZSpeed() / Z_STEPS_PER_MM) * JOG_LOOKAHEAD_S;
 float target = zPositionMM() + jogZ.dir * lookMM;
 stepperZA.moveTo(zStepsForMM(target));
 stepperZB.moveTo(zStepsForMM(target));
}

void refreshGripJog() {
 if (jogG.dir > 0 && gripForceExceeded()) {
 stopGripJog();
 return;
 }
 float lookMM = (jogGripSpeed() / GRIP_STEPS_PER_MM) * JOG_LOOKAHEAD_S;
 long lookSteps = (long)(lookMM * GRIP_STEPS_PER_MM);
 long target = stepperGrip.currentPosition() + GRIP_TIGHTEN_SIGN * jogG.dir * lookSteps;
 stepperGrip.moveTo(target);
}

// stop() sets the target to the nearest position reachable with a normal
// deceleration ramp, so run() brings it down smoothly rather than dropping
// the step train instantly.
void stopZJog() {
 if (jogZ.continuous) {
 stepperZA.stop();
 stepperZB.stop();
 }
 jogZ.continuous = false;
 jogZ.dir = 0;
 if (jogZ.speedOverridden) {
 stepperZA.setMaxSpeed(Z_MAX_SPEED_STEPS_PER_SEC);
 stepperZB.setMaxSpeed(Z_MAX_SPEED_STEPS_PER_SEC);
 jogZ.speedOverridden = false;
 }
}

void stopGripJog() {
 if (jogG.continuous) {
 stepperGrip.stop();
 }
 jogG.continuous = false;
 jogG.dir = 0;
 if (jogG.speedOverridden) {
 stepperGrip.setMaxSpeed(GRIP_MAX_SPEED_STEPS_PER_SEC);
 jogG.speedOverridden = false;
 }
}

void jogZDiscrete(int dir) {
 float stepMM = jogFast ? Z_JOG_STEP_FAST_MM : Z_JOG_STEP_FINE_MM;
 float target = zTargetMM() + dir * stepMM;
 stepperZA.moveTo(zStepsForMM(target));
 stepperZB.moveTo(zStepsForMM(target));
 Serial.print(F("JOG Z -> "));
 Serial.print(target, 2);
 Serial.println(F(" mm"));
}

void jogGripDiscrete(int dir) {
 if (dir > 0 && gripForceExceeded()) {
 return;
 }
 float stepMM = jogFast ? GRIP_JOG_STEP_FAST_MM : GRIP_JOG_STEP_FINE_MM;
 long delta = (long)(GRIP_TIGHTEN_SIGN * dir * stepMM * GRIP_STEPS_PER_MM);
 long target = stepperGrip.targetPosition() + delta;
 stepperGrip.moveTo(target);
 Serial.print(F("JOG GRIP -> opening "));
 Serial.print(openingMMForSteps(target), 2);
 Serial.println(F(" mm"));
}

bool gripForceExceeded() {
 if (!LOADCELL_ENABLED[0]) {
 return false;
 }
 if (cellRT[0].filtered >= GRIP_FORCE_LIMIT_G) {
 Serial.print(F("WARN grip force limit "));
 Serial.print(cellRT[0].filtered, 1);
 Serial.println(F(" g -- tightening blocked"));
 return true;
 }
 return false;
}

// ---------------- COMMAND DISPATCH ----------------

void processCommand(const String& message) {
 if (message.length() == 0) {
 return;
 }

 if (message == "JOG") {
 enterJogMode();

 } else if (message.startsWith("JOGLIVE ")) {
 // Browser press/hold. JOGLIVE Z UP|DOWN|STOP, JOGLIVE GRIP OPEN|CLOSE|STOP,
 // JOGLIVE PING (watchdog refresh for whichever axes are live).
 String arg = message.substring(8);
 arg.trim();
 if (arg == "Z UP") startLiveJogZ(-1);
 else if (arg == "Z DOWN") startLiveJogZ(+1);
 else if (arg == "Z STOP") stopLiveJogZ();
 else if (arg == "GRIP OPEN") startLiveJogGrip(-1);
 else if (arg == "GRIP CLOSE") startLiveJogGrip(+1);
 else if (arg == "GRIP STOP") stopLiveJogGrip();
 else if (arg == "PING") {
 unsigned long now = millis();
 if (liveJogZDir != 0) liveJogZLastMS = now;
 if (liveJogGripDir != 0) liveJogGripLastMS = now;
 } else {
 sendErr("usage: JOGLIVE Z UP|DOWN|STOP | GRIP OPEN|CLOSE|STOP | PING");
 }

 } else if (message.startsWith("Z ")) {
 String arg = message.substring(2);
 arg.trim();
 float targetMM;
 if (!parseFloatStrict(arg, targetMM)) {
 sendErr("bad value");
 return;
 }
 moveZTo(targetMM);

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

 } else if (message == "GRIP?" || message == "POS?") {
 printPosition();

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
 } else if (arg[0] == '+' || arg[0] == '-') {
 float deltaMM;
 if (!parseFloatStrict(arg, deltaMM)) {
 sendErr("bad value");
 return;
 }
 moveGripRelative(deltaMM);
 } else {
 sendErr("grip needs OPEN, CLOSE <width>, or a signed delta like +10");
 }

 } else if (message == "STOP") {
 stopZJog();
 stopGripJog();
 stopLiveJogZ();
 stopLiveJogGrip();
 stepperZA.stop();
 stepperZB.stop();
 stepperSelect.stop();
 stepperGrip.stop();
 Serial.println(F("STOPPING"));

 } else if (message == "RATE?") {
 Serial.print(F("LOOP "));
 Serial.print(loopHz);
 Serial.println(F(" Hz"));

 } else if (message == "F?") {
 printForceLine();

 } else if (message == "CAL?") {
 calPrint();

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

 } else if (message.startsWith("SETTLE ")) {
 String arg = message.substring(7);
 arg.trim();
 if (arg == "ON") {
 for (int i = 0; i < LOADCELL_COUNT; i++) {
 if (!LOADCELL_ENABLED[i]) continue;
 cellRT[i].settle = SETTLE_IDLE;
 cellRT[i].stableUpdates = 0;
 }
 Serial.println(F("SETTLE ARMED"));
 } else if (arg == "OFF") {
 for (int i = 0; i < LOADCELL_COUNT; i++) {
 cellRT[i].settle = SETTLE_OFF;
 }
 Serial.println(F("SETTLE OFF"));
 } else {
 sendErr("usage: SETTLE ON | SETTLE OFF");
 }

 } else if (message.startsWith("MARK")) {
 // Drops a labelled row into the log so you can find trial boundaries.
 String label = message.substring(4);
 label.trim();
 Serial.print(F("MARK "));
 Serial.print(millis());
 Serial.print(' ');
 Serial.println(label);

 } else if (message.startsWith("STREAM")) {
 String arg = message.substring(6);
 arg.trim();
 if (arg.startsWith("ON")) {
 String msArg = arg.substring(2);
 msArg.trim();
 if (msArg.length() > 0) {
 float ms;
 if (parseFloatStrict(msArg, ms) && ms >= 5) {
 streamIntervalMS = (unsigned long)ms;
 } else {
 sendErr("stream interval must be >= 5 ms");
 return;
 }
 }
 streaming = true;
 Serial.println(F("# t_ms g1 N1 g2 N2 z_mm grip_mm"));
 Serial.print(F("STREAM ON every "));
 Serial.print(streamIntervalMS);
 Serial.println(F(" ms"));
 } else if (arg == "OFF") {
 streaming = false;
 Serial.println(F("STREAM OFF"));
 } else {
 sendErr("usage: STREAM ON [interval_ms] | STREAM OFF");
 }

 } else if (message == "HELP") {
 printHelp();

 } else {
 sendErr("unknown command");
 }
}

// ---------------- MOTION ----------------

long zStepsForMM(float mm) {
 return (long)(Z_DOWN_SIGN * mm * Z_STEPS_PER_MM);
}

float zPositionMM() {
 return (float)stepperZA.currentPosition() / (Z_DOWN_SIGN * Z_STEPS_PER_MM);
}

float zTargetMM() {
 return (float)stepperZA.targetPosition() / (Z_DOWN_SIGN * Z_STEPS_PER_MM);
}

void moveZTo(float targetMM) {
 stepperZA.moveTo(zStepsForMM(targetMM));
 stepperZB.moveTo(zStepsForMM(targetMM));
 zMovePending = true;

 Serial.print(F("MOVING Z to "));
 Serial.print(targetMM, 2);
 Serial.println(F(" mm below start"));
}

void moveSelectTo(long targetSteps) {
 long clamped = clampSteps(targetSteps, SELECT_MIN_STEPS, SELECT_MAX_STEPS, "SELECT");
 stepperSelect.moveTo(clamped);
 selectMovePending = true;
}

void moveGripToSteps(long targetSteps) {
 stepperGrip.moveTo(targetSteps);
 gripMovePending = true;
}

// Relative jog. Positive deltaMM tightens by that many mm of jaw travel,
// negative loosens. Works off the current TARGET rather than the mid-flight
// position, so several commands fired back to back stack correctly.
void moveGripRelative(float deltaMM) {
 long base = stepperGrip.targetPosition();
 long deltaSteps = (long)(GRIP_TIGHTEN_SIGN * deltaMM * GRIP_STEPS_PER_MM);

 moveGripToSteps(base + deltaSteps);

 Serial.print(F("MOVING GRIP "));
 Serial.print(deltaMM, 2);
 Serial.print(F(" mm -> target opening "));
 Serial.print(openingMMForSteps(stepperGrip.targetPosition()), 2);
 Serial.println(F(" mm"));
}

long gripStepsForWidth(float widthMM) {
 float travelMM = GRIP_MAX_OPENING_MM - widthMM;
 if (travelMM < 0) {
 travelMM = 0;
 }
 return GRIP_OPEN_STEPS + (long)(travelMM * GRIP_STEPS_PER_MM);
}

float openingMMForSteps(long steps) {
 return GRIP_MAX_OPENING_MM - ((float)(steps - GRIP_OPEN_STEPS) / GRIP_STEPS_PER_MM);
}

float gripOpeningMM() {
 return openingMMForSteps(stepperGrip.currentPosition());
}

void printPosition() {
 Serial.print(F("POS z="));
 Serial.print(zPositionMM(), 3);
 Serial.print(F(" mm grip="));
 Serial.print(gripOpeningMM(), 3);
 Serial.print(F(" mm f1="));
 Serial.print(cellRT[0].filtered, 1);
 Serial.print(F(" g f2="));
 Serial.print(cellRT[1].filtered, 1);
 Serial.print(F(" g moving="));
 Serial.println(anyAxisMoving() ? F("yes") : F("no"));
}

bool anyAxisMoving() {
 return stepperZA.distanceToGo() != 0 || stepperZB.distanceToGo() != 0 ||
 stepperSelect.distanceToGo() != 0 || stepperGrip.distanceToGo() != 0;
}

long clampSteps(long value, long minSteps, long maxSteps, const char* axisName) {
 long clamped = value;
 if (clamped < minSteps) {
 clamped = minSteps;
 } else if (clamped > maxSteps) {
 clamped = maxSteps;
 }
 if (clamped != value) {
 Serial.print(F("WARN "));
 Serial.print(axisName);
 Serial.print(F(" clamped to "));
 Serial.println(clamped);
 }
 return clamped;
}

// ---------------- LOAD CELLS ----------------

void serviceLoadCells() {
 for (int i = 0; i < LOADCELL_COUNT; i++) {
 if (!LOADCELL_ENABLED[i]) continue;

 if (loadCell[i]->update()) {
 float g = loadCell[i]->getData();
 CellRuntime& rt = cellRT[i];

 if (!rt.haveFirst) {
 rt.filtered = g;
 rt.previous = g;
 rt.baseline = g;
 rt.haveFirst = true;
 } else {
 rt.filtered += LOADCELL_ALPHA * (g - rt.filtered);
 }

 updateSettle(i);
 rt.previous = rt.filtered;
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
 runSteppers();
 }
}

// Detects an object being placed, waits for the reading to stop climbing, then
// reports the plateau force relative to the pre-placement baseline. Re-arms
// itself so repeated trials each print a line.
void updateSettle(int i) {
 CellRuntime& rt = cellRT[i];
 if (rt.settle == SETTLE_OFF) {
 return;
 }
 float step = rt.filtered - rt.previous;

 if (rt.settle == SETTLE_IDLE) {
 if (step >= PLACEMENT_DERIVATIVE) {
 rt.baseline = rt.previous;
 rt.stableUpdates = 0;
 rt.settle = SETTLE_CLIMBING;
 }
 return;
 }

 // SETTLE_CLIMBING
 if (fabs(step) <= STABLE_PEAK_LIMIT) {
 rt.stableUpdates++;
 } else {
 rt.stableUpdates = 0;
 }

 if (rt.stableUpdates >= REQUIRED_STABLE_UPDATES) {
 float grams = rt.filtered - rt.baseline;
 Serial.print(F("SETTLED "));
 Serial.print(i + 1);
 Serial.print(' ');
 Serial.print(grams, 2);
 Serial.print(F(" g "));
 Serial.print(grams * G_TO_NEWTON, 4);
 Serial.println(F(" N"));
 rt.settle = SETTLE_IDLE;
 rt.stableUpdates = 0;
 }
}

// F <t_ms> <g1> <N1> <g2> <N2> <z_mm> <grip_mm>. A disabled cell reports 0.
void printForceLine() {
 Serial.print(F("F "));
 Serial.print(millis());
 for (int i = 0; i < LOADCELL_COUNT; i++) {
 Serial.print(' ');
 Serial.print(cellRT[i].filtered, 2);
 Serial.print(' ');
 Serial.print(cellRT[i].filtered * G_TO_NEWTON, 4);
 }
 Serial.print(' ');
 Serial.print(zPositionMM(), 3);
 Serial.print(' ');
 Serial.println(gripOpeningMM(), 3);
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
// result gets picked up in serviceLoadCells(), so the steppers keep running.
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
 if (!USE_HARDCODED_CAL) {
 Serial.println(F("WARN no stored calibration -- taring now; then run CAL <1|2> <grams>"));
 }
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

// ---------------- UTIL ----------------

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
 Serial.print(F("ERR "));
 Serial.println(reason);
}

void printHelp() {
 Serial.println(F("JOG keyboard control mode (q to exit)"));
 Serial.println(F("JOGLIVE ... browser press/hold jog (see gantry-jog-panel.html)"));
 Serial.println(F("Z <mm> move Z to <mm> below start"));
 Serial.println(F("SELECT GEKKO rotate turret to head A"));
 Serial.println(F("SELECT SILICONE rotate turret to head B"));
 Serial.println(F("GRIP OPEN jaws to fully-open home"));
 Serial.println(F("GRIP CLOSE <mm> close until jaw gap = object width"));
 Serial.println(F("GRIP +10 / GRIP -5 relative: + tightens, - loosens"));
 Serial.println(F("POS? z, grip, and both forces"));
 Serial.println(F("STOP decelerate everything now"));
 Serial.println(F("F? one force line"));
 Serial.println(F("STREAM ON [ms] continuous log lines (default 50 ms)"));
 Serial.println(F("STREAM OFF stop streaming"));
 Serial.println(F("MARK <text> drop a labelled row into the log"));
 Serial.println(F("TARE / TARE 1 re-zero the cells"));
 Serial.println(F("CAL 1 <grams> compute cal factor from known mass"));
 Serial.println(F("CAL? show current calibration"));
 Serial.println(F("SETTLE ON / OFF auto-report plateau force on placement"));
 Serial.println(F("RATE? loop frequency, for diagnosing stalls"));
}
