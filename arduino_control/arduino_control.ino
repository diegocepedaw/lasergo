#include <Servo.h>
Servo servoX;   
Servo servoY;   
String inByte;
int pos;
int whichservo = 1;
int servoXPos = 0;
int servoYPos = 0;

void setup() {
  servoX.attach(9);
  servoY.attach(8);
  Serial.begin(9600);
  servoX.write(servoXPos);
  servoY.write(servoYPos);  
}

void loop()
{    
  if(Serial.available())  // if data available in serial port
    {
    while (Serial.available() == 0) {
      // Wait for User to Input Data
    }
    inByte = Serial.readString(); // read data until newline
    String str_data = inByte;
    int firstCommaIndex = str_data.indexOf(",");
    int secondCommaIndex = str_data.indexOf(",", firstCommaIndex+1);
    String cmd = str_data.substring(0, firstCommaIndex);
    String param1 = str_data.substring(firstCommaIndex+1, secondCommaIndex);
    String param2 = str_data.substring(secondCommaIndex+1);
    //Serial.println(str_data + "," +cmd + "," + param1 + "," + param2);

    // python code will gove increments translate it into position
    pos = servoXPos + param1.toInt();       
    servoX.write(pos);     // move servo
    Serial.print("servo in position: ");  
    servoXPos = pos;

    pos = servoYPos + param2.toInt();       
    servoY.write(pos);     // move servo
    servoYPos = pos;
    Serial.print("Servo positions: ");  
    Serial.println(String(servoXPos)+","+String(servoYPos));
  
    }
}
