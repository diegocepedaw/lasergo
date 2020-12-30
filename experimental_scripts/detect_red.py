import cv2
import numpy as np
import time
import math
import pickle



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


def target_laser(target, cap, sel):
    while (1):
        begin = time.time()
        # Take each frame
        ret, frame = cap.read()
        frame = image_resize(frame, maxLength = 720, inter = cv2.INTER_AREA)
        fill_color = [0, 0, 0]
        frame[sel] = fill_color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0, 70, 50])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 70, 50])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)
        cv2.imshow('mask1', mask1)
        cv2.imshow('mask2', mask2)


        thresh1 = cv2.threshold(mask1, 25, 255, cv2.THRESH_BINARY)[1]
        thresh2 = cv2.threshold(mask2, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.bitwise_or(thresh1, thresh1, mask=thresh2)
        lel = cv2.findNonZero(mask2)
        if lel is not None:
            if len(lel) > 15:
                lel = lel[-10:]
        if time.time() - begin > 4:
            continue
        x = 0
        y = 0
        if lel is not None:
            for element in lel:
                y += element[0][0]
                x += element[0][1]
            x = x / len(lel)
            y = y / len(lel)

        laser_coords = (int(y),int(x))
        distance_to_target = math.sqrt( ((laser_coords[0]-target[0])**2)+((laser_coords[1]-target[1])**2) )
        if distance_to_target < 20:
            cv2.circle(frame, target, 5, (0, 255, 0), -1)
        cv2.circle(frame, target, 10, (255, 0, 0), 2)


        cv2.imshow('mask', thresh)
        cv2.imshow('Track Laser', frame)
        time.sleep(0.5)
        waitkey = cv2.waitKey(1)
        if waitkey & 0xFF == ord('q') or waitkey & 0xFF == ord('q') or waitkey == 9:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
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
    target_laser((300,200), cap, sel)
    cap.release()