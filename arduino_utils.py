import cv2
import numpy as np
import time
import math
import serial
import pickle

LASER_START = (100,100)
arduino = serial.Serial('COM6', 9600, timeout=5)

def clear_leds():
    # turn off all leds
    data = bytes("C0,0\r\n", "utf8")
    arduino.write(data)                          # write increment to serial port 
    print("wrote: " + str(data)) 
    reachedPos = str(arduino.readline())            # read serial port for arduino echo
    print("read: " + str(reachedPos))

def set_led_coordinates(x_coord,y_coord):
    y_coord += 19
    # light up the leds on the board to indicate a coordinate
    data = bytes("O" + str(x_coord) + "," + str(y_coord)+'\r\n', 'utf8')
    arduino.write(data)                          # write position to serial port
    print("wrote: " + str(data)) 
    reachedPos = str(arduino.readline())            # read serial port for arduino echo
    print("read: " + str(reachedPos))

def control_laser(x_travel,y_travel):
    data = bytes("I" + str(x_travel) + "," + str(y_travel)+'\r\n', 'utf8')
    arduino.write(data)                          # write increment to serial port 
    reachedPos = str(arduino.readline())            # read serial port for arduino echo
    time.sleep(0.5)
    print(reachedPos)

def set_laser_pos(x_travel,y_travel):
    data = bytes("S" + str(x_travel) + "," + str(y_travel)+'\r\n', 'utf8')
    arduino.write(data)                          # write position to serial port 
    reachedPos = str(arduino.readline())            # read serial port for arduino echo
    time.sleep(1)
    print(reachedPos)
            

def calibrate_laser(cap):
    ''' determine how many pixel sevo angle travels'''
    set_laser_pos(LASER_START[0],LASER_START[1])
    ret, frame = cap.read()
    frame = image_resize(frame, maxLength = 720, inter = cv2.INTER_AREA)
    laser_coords = get_laser_coords(frame)
    print(laser_coords)
    # move laser by 1 pos and measure change in pixel x,y
    control_laser(3,3)
    ret, frame = cap.read()
    new_laser_coords = get_laser_coords(frame)
    x_ratio = (new_laser_coords[0] - laser_coords[0])
    y_ratio = (new_laser_coords[1] - laser_coords[1])
    print(new_laser_coords)
    print("ratio:")
    print(x_ratio,y_ratio)
    return(x_ratio,y_ratio)




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
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # in the HSV range red is split up in two parts so these masks capture different red values which work under different conditions
    # currently I am just manually setting one but in the future this should be done in a better way
    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 70, 50])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    mask = mask2
    

    thresh = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("mask", thresh)
    lel = cv2.findNonZero(thresh)
    if lel is not None:
        if len(lel) > 500:
            lel = lel[-450:]
    x = 0
    y = 0
    if lel is not None:
        for element in lel:
            x += element[0][0]
            y += element[0][1]
        x = x / len(lel)
        y = y / len(lel)

    laser_coords = (int(x),int(y))
    #cv2.circle(frame, laser_coords, 10, (0, 0, 0), 2)

    return laser_coords

def target_laser(target, cap, mask = True):
    x_ratio, y_ratio = calibrate_laser(cap)
    x_ratio, y_ratio = -35, -14
   
    ret, frame = cap.read()
    frame = image_resize(frame, maxLength = 720, inter = cv2.INTER_AREA)
    with open('corner_original_coords.data', 'rb') as filehandle:
        # read the data as binary data stream
        pts1 = pickle.load(filehandle)
        #arrange points to be drawn as polygon
        poly = [pts1[0], pts1[2], pts1[3], pts1[1]]
        # MASK NON BOARD AREA
        fill_color = [0, 0, 0] # any BGR color value to fill with
        mask_value = 255            # 1 channel white (can be any non-zero uint8 value)
        # our stencil - some `mask_value` contours on black (zeros) background, 
        # the image has same height and width as `img`, but only 1 color channel
        stencil  = np.zeros(frame.shape[:-1])
        cv2.fillPoly(stencil,  np.array([poly], dtype=np.int32), mask_value)
        sel = stencil != mask_value # select everything that is not mask_value
        arrived = False
    while (1):
        ret, frame = cap.read()
        frame = image_resize(frame, maxLength = 720, inter = cv2.INTER_AREA)
        frame[sel] = fill_color            # fill masked area with fill_color
        laser_coords = get_laser_coords(frame)
        print(laser_coords)
        cv2.circle(frame, target, 10, (0, 0, 0), 2)
        x_dist = abs(target[0] - laser_coords[0])
        y_dist = abs(target[1] - laser_coords[1])
        if  x_dist < 8 and y_dist < 8 or arrived:
            arrived = True
            cv2.circle(frame, target, 17, (0, 255, 0), 2)
        else:
            x_travel = ((target[0] - laser_coords[0] ) / x_ratio)
            y_travel = ((target[1] - laser_coords[1] ) / y_ratio)
            if x_dist < 5:
                x_travel = 0
            elif x_travel < 0:
                x_travel = math.floor(x_travel)
            else:
                x_travel = math.ceil(x_travel)
            if y_dist < 5:
                 y_travel = 0
            elif y_travel < 0:
                y_travel = math.floor(y_travel)
            else:
                y_travel = math.floor(y_travel)
            print(laser_coords, target)
            print(x_travel,y_travel)
            time.sleep(3)
            control_laser(x_travel,y_travel)



        #cv2.imshow('mask', mask)
        cv2.imshow('Track Laser', frame)
        waitkey = cv2.waitKey(1)
        if waitkey & 0xFF == ord('q') or waitkey & 0xFF == ord('q') or waitkey == 9:
            break
    set_laser_pos(0,0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    #calibrate_laser(cap)
    target_laser((340, 200), cap)
    cap.release()
    if arduino.isOpen() == True:
        arduino.close()