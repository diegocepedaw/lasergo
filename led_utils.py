import cv2
import time
import serial
from random import randrange
arduino = serial.Serial('COM6', 9600, timeout=5)

def turn_off_leds(x_travel,y_travel):
    # turn off all leds
    data = bytes("I0,0\r\n", "utf8")
    arduino.write(data)                          # write increment to serial port 
    reachedPos = str(arduino.readline())            # read serial port for arduino echo

    print(reachedPos)

def set_led_coordinates(x_coord,y_coord):
    # light up the leds on the board to indicate a coordinate
    data = bytes("O" + str(x_coord) + "," + str(y_coord)+'\r\n', 'utf8')
    arduino.write(data)                          # write position to serial port
    print("wrote: " + str(data)) 
    reachedPos = str(arduino.readline())            # read serial port for arduino echo
    print("read: " + str(reachedPos))
            
if __name__ == "__main__":
    # light up random coordinate every second
    while(True):
        print("Input x coord:")
        x = int(input())
        print("Input x coord:")
        y = int(input())
        set_led_coordinates(x,y+19)
 
    if arduino.isOpen() == True:
        arduino.close()