import cv2
import numpy as np
from gridDetect import show_wait_destroy
import blend_modes
from builtins import input
from image_processing_utils import apply_processing

def start_capture():
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "captured_images/opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
            break

    cam.release()

    cv2.destroyAllWindows()

def capture_image(cam):
    ''' capture an image from the camera and apply procesing '''
    # manually set focus
    focus = 0  # min: 0, max: 255, increment:5
    cam.set(28, focus) 
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        return
    frame = apply_processing(frame)
    return frame


if __name__ == "__main__":
  cam = cv2.VideoCapture(0)
  capture_image(cam)