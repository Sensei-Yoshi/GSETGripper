const unsigned long BAUD_RATE = 9600;

String command;

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  Serial.begin(BAUD_RATE);
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
}

void processCommand(const String& message) {
  if (message == "LED_ON") {
    digitalWrite(LED_BUILTIN, HIGH);
  } else if (message == "LED_OFF") {
    digitalWrite(LED_BUILTIN, LOW);
  }
}
