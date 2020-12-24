# servo control 15.12.2016
 
# 1) user set servo position in python
# 2) position is sent to arduino
# 3) arduino moves servo to position
# 4) arduino send confirmation message back to python
# 5) message is printed in python console
 
import serial                                           # import serial library
arduino = serial.Serial('COM5', 9600)   # create serial object named arduino
while True:                                             # create loop
 
        command = bytes(input ("Servo position: "), 'utf8')       # query servo position
        print(command)
        if command == bytes("q", "utf-8"):
            print("exiting...")
            break
        arduino.write(command)                          # write position to serial port
        reachedPos = str(arduino.readline())            # read serial port for arduino echo
        print(reachedPos)                               # print arduino echo to console
if arduino.isOpen() == True:
        arduino.close()