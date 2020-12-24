import cv2
import numpy as np
from gridDetect import show_wait_destroy
import blend_modes
from builtins import input
from image_processing_utils import apply_processing

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