"""
@file morph_lines_detection.py
@brief Use morphology transformations for extracting horizontal and vertical lines sample code
"""
import numpy as np
import sys
import cv2 as cv
import imutils

def show_wait_destroy(winname, img):
    cv.imshow(winname, img)
    cv.moveWindow(winname, 600, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)
def main(argv):
    # [load_image]
    # Check number of arguments
    if len(argv) < 1:
        print ('Not enough parameters')
        print ('Usage:\nmorph_lines_detection.py < path_to_image >')
        return -1
    # Load the image
    src = cv.imread(argv[0], cv.IMREAD_COLOR)
    src = cv.resize(src, (570,570), interpolation = cv.INTER_AREA)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image: ' + argv[0])
        return -1
    # Show source image
    #cv.imshow("src", src)
    # [load_image]
    # [gray]
    # Transform source image to gray if it is not already
    if len(src.shape) != 2:
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    else:
        gray = src
    # Show gray image
    #show_wait_destroy("gray", gray)
    # [gray]
    # [bin]
    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    gray = cv.bitwise_not(gray)
    bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                                cv.THRESH_BINARY, 15, -2)
    # Show binary image
    #show_wait_destroy("binary", bw)
    # [bin]
    # [init]
    # Create the images that will use to extract the horizontal and vertical lines
    horizontal = np.copy(bw)
    vertical = np.copy(bw)

    # [init]
    # [horiz]
    # Specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = cols // 40
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
    # Apply morphology operations
    horizontal = cv.erode(horizontal, horizontalStructure)
    horizontal = cv.dilate(horizontal, horizontalStructure)
    # Show extracted horizontal lines
    #show_wait_destroy("horizontal", horizontal)

    # [horiz]
    # [vert]
    # Specify size on vertical axis
    rows = vertical.shape[0]
    verticalsize = rows // 40
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
    # Apply morphology operations
    vertical = cv.erode(vertical, verticalStructure)
    vertical = cv.dilate(vertical, verticalStructure)
    
    # Show extracted vertical lines
    #show_wait_destroy("vertical", vertical)

    bwv = cv.adaptiveThreshold(vertical, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                                cv.THRESH_BINARY, 15, -2)
    bwh = cv.adaptiveThreshold(horizontal, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                                cv.THRESH_BINARY, 15, -2)
    v_h_and = cv.bitwise_and(bwv,bwh)

    #show_wait_destroy("and", v_h_and)

    ### finding contours, can use connectedcomponents aswell
    contours,_ = cv.findContours(v_h_and, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    ### converting to bounding boxes from polygon
    contours=[cv.boundingRect(cnt) for cnt in contours]
    img = np.zeros((src.shape[0],src.shape[1],3), np.uint8)
    ### drawing rectangle for each contour for visualising
    for cnt in contours:
        x,y,w,h=cnt
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),10)
    #show_wait_destroy("and", img)

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    contours,_ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    ### converting to bounding boxes from polygon
    contours=[cv.boundingRect(cnt) for cnt in contours]
    img = np.zeros((src.shape[0],src.shape[1],3), np.uint8)
    ### drawing rectangle for each contour for visualising
    for cnt in contours:
        x,y,w,h=cnt
        #cv.rectangle(src,(x,y),(x+w,y+h),(0,255,0),2)

        center_coords = (int((x+w/2)), int((y+h/2)))
        # make a 60px square cocentric with the contour
        # this represents each individual area that will be evaulated
        cv.rectangle(src,(center_coords[0]-15,center_coords[1]-15),(center_coords[0]+15,center_coords[1]+15),(128,0,128),2)
        cv.circle(src, center_coords, 3, (255, 255, 255), -1)
    
    show_wait_destroy("intersections", src)

    filename = 'gridfiles/evaluation_grid.jpg'
    cv.imwrite(filename, src) 
    
    return 0
if __name__ == "__main__":
    main(sys.argv[1:])