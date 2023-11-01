#include <Servo.h>

Servo myservo1;
Servo myservo2;

int mouthPin=5;
int bluePin=6;
int redPin=7;

int mouthV=0;
int eyeV=1;

int posx=1500;
int posy=1500;

const int bufferSize = 32;
char inputBuffer[bufferSize];
int receivedInts[4];

bool isValidMicroseconds(int pos) {
  return (pos >= 500 && pos <= 2600); // validate that the position is within an acceptable range
}

void setup() {
  myservo1.attach(3);
  myservo2.attach(11);
  
  pinMode(mouthPin,OUTPUT);
  pinMode(bluePin,OUTPUT);  
  pinMode(redPin,OUTPUT);
  
  digitalWrite(mouthPin,LOW);
  digitalWrite(bluePin,HIGH);
  digitalWrite(redPin,HIGH);

  Serial.begin(115200);
}

void loop() {
     if (Serial.available()) {
        // Read the incoming message until a newline is encountered
        Serial.readBytesUntil('\n', inputBuffer, bufferSize);
        
        // Parse the integers
        sscanf(inputBuffer, "%d,%d,%d,%d", &receivedInts[0], &receivedInts[1], &receivedInts[2], &receivedInts[3]);
    
        // For demonstration, let's echo back the received integers
        Serial.print("Received ints: ");
        for (int i = 0; i < 4; i++) {
          Serial.print(receivedInts[i]);
          Serial.print(" ");
        }
        posx=receivedInts[0];
        posy=receivedInts[1];
        mouthV=receivedInts[2];
        eyeV=receivedInts[3];
        Serial.println();
    
        // Clear the inputBuffer
        memset(inputBuffer, 0, bufferSize);
        
        // Write mouth open or closed
        if (mouthV==1){
            digitalWrite(mouthPin,HIGH);
        }else{
            digitalWrite(mouthPin,LOW);
        }
        
        // Write eye color
        if (eyeV==1){
            digitalWrite(redPin,LOW);
            digitalWrite(bluePin,HIGH);
        }
        if (eyeV>=2){
            digitalWrite(redPin,HIGH);
            digitalWrite(bluePin,LOW);
        }
        if (eyeV==0){
            digitalWrite(redPin,HIGH);
            digitalWrite(bluePin,HIGH);
        }

        // Write to the servo         
        if (isValidMicroseconds(posx) && isValidMicroseconds(posy)) {
            myservo1.writeMicroseconds(posx);
            myservo2.writeMicroseconds(posy);
        } else {
            Serial.println("Invalid values received");
        }
    }
}
