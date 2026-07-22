#include <Arduino.h>



void setup() {
  // put your setup code here, to run once:
   pinMode(13, OUTPUT);
   pinMode(12, OUTPUT);
   digitalWrite(13, HIGH);
   delay(10000);
   digitalWrite(13,LOW);
   digitalWrite(12, HIGH);
   delay(10000);
   digitalWrite(12, LOW);
}

void loop() {
  
 }

