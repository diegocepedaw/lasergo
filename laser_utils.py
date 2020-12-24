import cv2
import numpy as np
import time
import math
import serial

LASER_START = (40,40)
arduino = serial.Serial('COM5', 9600)

def control_laser(x_travel,y_travel):
    data = bytes(str(y_travel) + "," + str(x_travel)+'\r\n', 'utf8')
    arduino.write(data)                          # write position to serial port 
    time.sleep(1)
    reachedPos = str(arduino.readline())            # read serial port for arduino echo
    print(reachedPos)
            

def calibrate_laser(cam):
    ''' determine how many pixel sevo angle travels'''
    control_laser(LASER_START[0],LASER_START[1])
    ret, frame = cap.read()
    frame = image_resize(frame, maxLength = 720, inter = cv2.INTER_AREA)
    laser_coords = get_laser_coords(frame)
    print(laser_coords)
    # move laser by 1 pos and measure change in pixel x,y
    control_laser(1,1)
    ret, frame = cap.read()
    new_laser_coords = get_laser_coords(frame)
    x_ratio = new_laser_coords[0] - laser_coords[0]
    y_ratio = new_laser_coords[1] - laser_coords[1]
    print(new_laser_coords) 
    print(x_ratio,y_ratio)




def image_resize(image, maxLength = 720, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # check to see if height is larger than width
    if max(h, w) == h:
        # calculate the ratio of the height and construct the
        # dimensions
        r = maxLength / float(h)
        dim = (int(w * r), maxLength)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = maxLength / float(w)
        dim = (maxLength, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def get_laser_coords(frame):
    print("here")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 0, 255])
    upper_red = np.array([255, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    thresh = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)[1]
    lel = cv2.findNonZero(thresh)
    if lel is not None:
        if len(lel) > 15:
            lel = lel[-10:]
    x = 0
    y = 0
    if lel is not None:
        for element in lel:
            y += element[0][0]
            x += element[0][1]
        x = x / len(lel)
        y = y / len(lel)

    laser_coords = (int(y),int(x))
    return laser_coords

def target_laser(target, cap):
    while (1):
        ret, frame = cap.read()
        frame = image_resize(frame, maxLength = 720, inter = cv2.INTER_AREA)
        laser_coords = get_laser_coords(frame)
        distance_to_target = math.sqrt( ((laser_coords[0]-target[0])**2)+((laser_coords[1]-target[1])**2) )
        if distance_to_target < 20:
            cv2.circle(frame, target, 5, (0, 255, 0), -1)
        cv2.circle(frame, target, 10, (255, 0, 0), 2)


        cv2.imshow('mask', mask)
        cv2.imshow('Track Laser', frame)
        time.sleep(0.5)
        waitkey = cv2.waitKey(1)
        if waitkey & 0xFF == ord('q') or waitkey & 0xFF == ord('q') or waitkey == 9:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    calibrate_laser(cap)
    #   target_laser((300,200), cap)
    cap.release()
    if arduino.isOpen() == True:
        arduino.close()