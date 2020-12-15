"""
@file morph_lines_detection.py
@brief Use morphology transformations for extracting horizontal and vertical lines sample code
"""
import numpy as np
import sys
import cv2 as cv
import imutils
import pickle
import uuid

def show_wait_destroy(winname, img):
    cv.imshow(winname, img)
    cv.moveWindow(winname, 600, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)
def  process_analysis_grid(squareFile):
    ''' process and image and extract coordinates for every intersection on the board'''

    # [load_image]
    # Check number of arguments
    # Load the image
    src = cv.imread(squareFile, cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image: ' + squareFile)
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
    transparent_background = np.zeros((570, 580, 4))
    coord_list =[]

    # remove any extra points by searching for points that have too close a neighbor and removing them
    for cnt in contours:
        x,y,w,h=cnt
        #cv.rectangle(src,(x,y),(x+w,y+h),(0,255,0),2)

        center_coords = (int((x+w/2)), int((y+h/2)))
        coord_list.append(center_coords)
        cv.circle(transparent_background, center_coords, 5, (255, 255, 255), -1)
    show_wait_destroy("intersections", transparent_background)

    # remove any extra points by searching for points that have too close a neighbor and removing them
    pared_coords = []
    # sort coordinates by x axis
    coord_list_i = sorted(coord_list, key = lambda x: x[0])
    coord_listy = sorted(coord_list, key = lambda x: x[0])
    rejected_coords = set()
    print(coord_list)
    for icoord in coord_list_i:
        for jcoord in coord_listy:
            if icoord == jcoord or jcoord in rejected_coords:
                continue
            if abs(icoord[0] - jcoord[0]) < 20:
                if abs(icoord[1] - jcoord[1]) < 20:
                    print(icoord,jcoord)
                    coord_list_i.remove(jcoord)
                    coord_listy.remove(jcoord)
                    rejected_coords.add(jcoord)
                    break
        pared_coords.append(icoord)
        
    print(len(contours))
    print(len(pared_coords))

    grid_background = np.zeros((570, 580, 4))
    for coord in pared_coords:
        x,y = coord
        #cv.rectangle(src,(x,y),(x+w,y+h),(0,255,0),2)

        # make a 60px square cocentric with the contour
        # this represents each individual area that will be evaulated
        cv.rectangle(src,(x-15,y-15),(x+15,y+15),(128,0,128),2)
        cv.circle(src, coord, 3, (255, 255, 255), -1)

        cv.circle(transparent_background, coord, 5, (255, 0, 255), -1)
        cv.circle(grid_background, coord, 5, (255, 255, 255), -1)
    
    show_wait_destroy("intersections", src)
    show_wait_destroy("intersections", transparent_background)
    

    filename = 'gridfiles/evaluation_grid.png'
    cv.imwrite(filename, grid_background) 
    
    return pared_coords

def crop_and_save(src_file, out_path):
    ''' crop a board into images of each individual intersection and save them to disk to create the training dataset'''
    # open image
    with open('grid_coords.data', 'rb') as filehandle:
        # read the data as binary data stream
        grid_coords = pickle.load(filehandle)

    src = cv.imread(src_file)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image: ' + squareFile)
        return -1

    for coord in grid_coords:
        x,y = coord
        ymin = y-15 if y > 15 else 0
        ymax = y+15 if y < 585 else 600
        xmin = x-15 if x > 15 else 0
        xmax = x+15 if x < 585 else 600
        crop_img = src[ymin:ymax, xmin:xmax]
        #show_wait_destroy("crop", crop_img)
        cv.imwrite(out_path+str(uuid.uuid4())+".jpg", crop_img) 

    for coord in grid_coords:
        x,y = coord
        ymin = y-15 if y > 15 else 0
        ymax = y+15 if y < 585 else 600
        xmin = x-15 if x > 15 else 0
        xmax = x+15 if x < 585 else 600
        # make a 60px square cocentric with the contour
        cv.rectangle(src,(xmin,ymin),(xmax,ymax),(128,0,128),2)
    show_wait_destroy("intersections", src)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        crop_and_save(sys.argv[1], sys.argv[2])
    else: 
        process_analysis_grid(sys.argv[1:][0])