#here is code to check the output
import cv2
import numpy as np
import webcolors
import os
from math import sqrt



#######################
#try straightening again with te top 4 detected corners
#######################
# def rgb_to_hex(rgb):
#     return '%02x%02x%02x' % rgb


COLORS = (
    (205,226,228),
    (79,76,61),
    (237,168,82)
)

WHITE = (205,226,228)
BLACK = (79,76,61)
BOARD = (237,168,82)


def closest_color(rgb):
    r, g, b = rgb
    color_diffs = []
    for color in COLORS:
        cr, cg, cb = color
        color_diff = sqrt(abs(r - cr)**2 + abs(g - cg)**2 + abs(b - cb)**2)
        color_diffs.append((color_diff, color))
    return min(color_diffs)[1]
    
def determine_color(src, answer):
    img = cv2.imread(src)###path of image file which we want to detect
    width, height, _ = img.shape 
    left = 5
    top = 5
    right = 5
    bottom = 5

    crop_img = img[5:height-5, 5:width-5]
    img = crop_img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("src", img)
    colors, count = np.unique(img.reshape(-1,img.shape[-1]), axis=0, return_counts=True)
    rgb = colors[count.argmax()]
    rgb = (rgb[0],rgb[1],rgb[2])
    color = closest_color(rgb)
    if(color == BLACK):
        processed_color ="black"
    elif(color == WHITE):
        processed_color ="white"
    else:
        processed_color = "board"

    return processed_color

def test_color_picker():
    # test black
    test_images = 100
    i = 0
    mistakes = 0
    f = r'training_images\black_stone'
    for file in os.listdir(f):
        f_img = f+"/"+file
        color = determine_color(f_img, "black")
        if color != "black":
            mistakes += 1
        
        #img = Image.open(f_img)
        if i > test_images:
            print("Black test:")
            print("correctly labeled: %s", str(i-mistakes))
            print("incorrectly labeled: %s", str(mistakes))
            break
        i += 1

    # test white
    i = 0
    mistakes = 0
    f = r'training_images\white_stone'
    for file in os.listdir(f):
        f_img = f+"/"+file
        color = determine_color(f_img, "white")
        if color != "white":
            mistakes += 1
        
        #img = Image.open(f_img)
        if i > test_images:
            print("White test:")
            print("correctly labeled: %s", str(i-mistakes))
            print("incorrectly labeled: %s", str(mistakes))
            break
        i += 1

    # test board
    i = 0
    mistakes = 0
    f = r'training_images\empty_point'
    for file in os.listdir(f):
        f_img = f+"/"+file
        color = determine_color(f_img, "board")
        if color != "board":
            mistakes += 1
        
        #img = Image.open(f_img)
        if i > test_images:
            print("Board test:")
            print("correctly labeled: %s", str(i-mistakes))
            print("incorrectly labeled: %s", str(mistakes))
            break
        i += 1



if __name__ == "__main__":
    test_color_picker()
    
