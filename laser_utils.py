import cv2
import numpy as np
import time
import math



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


def target_laser(target):
    cap = cv2.VideoCapture(0)
    while (1):
        begin = time.time()
        # Take each frame
        ret, frame = cap.read()
        frame = image_resize(frame, maxLength = 720, inter = cv2.INTER_AREA)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0, 0, 255])
        upper_red = np.array([255, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        
        thresh = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)[1]
        lel = cv2.findNonZero(thresh)
        if lel is not None:
            if len(lel) > 15:
                lel = lel[-10:]
        if time.time() - begin > 3:
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
        print(distance_to_target)
        if distance_to_target < 20:
            cv2.circle(frame, target, 5, (0, 255, 0), 2)
        cv2.circle(frame, target, 10, (255, 0, 0), 2)


        cv2.imshow('mask', mask)
        cv2.imshow('Track Laser', frame)
        time.sleep(0.5)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
  target_laser((300,200))