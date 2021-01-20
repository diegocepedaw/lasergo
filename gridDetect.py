"""
@file morph_lines_detection.py
@brief Use morphology transformations for extracting horizontal and vertical lines sample code
"""
import numpy as np
import sys
import cv2
import imutils
import pickle
import uuid
from scipy import spatial
from test_histogram_coorelation import evaluate_image
from sgf_utils import save_SGF

test_image = None
extra_coords = []
detected_coords = None
src_file = r"result\flatboard.jpg"
src_img = None

def draw_grid(img, line_color=(0, 255, 0), thickness=1, type_=cv2.LINE_AA, pxstep=40):
    '''(ndarray, 3-tuple, int, int) -> void
    draw gridlines on img
    line_color:
        BGR representation of colour
    thickness:
        line thickness
    type:
        8, 4 or cv2.LINE_AA
    pxstep:
        grid line frequency in pixels
    '''
    x = pxstep
    y = pxstep
    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += pxstep

    fitted_coords=[]
    board_size = 19
    i, j = 0, 0
    while i < board_size:
        while j < board_size:
            cv2.circle(img, (i*pxstep,j*pxstep), 15, (255, 0, 255), 2)
            j += 1
        j = 0
        i += 1

def transform_coords(coordinates):
    src = cv2.imread(src_file, cv2.IMREAD_COLOR)
    maxSide = 720
    src.shape
    pts1 = np.float32(coordinates)    
    pts2 = np.float32([[0, 0], [maxSide-1, 0], [0, maxSide-1], [maxSide-1, maxSide-1]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(src, matrix, (maxSide,maxSide))
    draw_grid(result)
    show_wait_destroy("transfor by corners", result)

def get_coorner_coords(coordinates):
    ''' this assumes that the top and bottom rows of points are calculated correctly'''
    boardize = 19
    # sort by y value
    coordinates = sorted(coordinates, key = lambda x: x[1])
    # sort the top row by x value
    top_left = sorted(coordinates[:19], key = lambda x: x[0])[0]
    top_right = sorted(coordinates[:19], key = lambda x: x[0])[18]
    bottom_left = sorted(coordinates[-19:], key = lambda x: x[0])[0]
    bottom_right = sorted(coordinates[-19:], key = lambda x: x[0])[18]
   
    return (top_left, top_right, bottom_left, bottom_right)

    

def map_to_goban_coordinates(coordinates):
    board_array = []
    line = []
    i = 1
    board_size=19
    coordinates = sorted(coordinates, key = lambda x: x[1])
    # map on goban coordinates 19x19
    for coord in coordinates:
        if i % (board_size) == 0:
            i = 1
            board_array.append(line)
            line.append((coord))
            line = []
            continue
        line.append((coord))
        
        i += 1
    
    sorted_board = []
    for row in board_array:
        sorted_board.append(sorted(row, key = lambda x: x[0]))
    return sorted_board

def closest_node(node, nodes):
    ''' return nearest neighbor to a coordanate in a list of coordinates'''
    tree = spatial.KDTree(nodes)
    return tree.query([node])[1][0]

def draw_circle(event, x, y, flags, param):
    global mouseX,mouseY
    global extra_coords
    global detected_coords
    extra_coords = []
    if event == cv2.EVENT_LBUTTONDOWN: 
        # cv2.circle(test_image, (x, y), 10, (255, 255, 255), -1)
        mouseX,mouseY = x,y
        extra_coords.append((x,y))
        #print(x,y)
    if event == cv2.EVENT_RBUTTONDOWN:
        coord_to_delete = closest_node((x,y), detected_coords)
        detected_coords.pop(coord_to_delete)
    draw_new_coords(test_image)

def draw_new_coords(img):
    global test_image
    global detected_coords
    global extra_coords
    global src_img

    detected_coords.extend(extra_coords) 
    #map_to_goban_coordinates(detected_coords)
    test_image = np.copy(src_img)
    for coord in detected_coords:
        x,y = coord
        cv2.circle(test_image, coord, 5, (255, 0, 255), -1)
        cv2.rectangle(test_image,(x-15,y-15),(x+15,y+15),(128,0,128),2)

def show_wait_destroy(winname, img):
    cv2.imshow(winname, img)
    cv2.moveWindow(winname, 600, 0)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)
def  process_analysis_grid(squareFile):

    global test_image
    global detected_coords
    global extra_coords
    global src_img
    ''' process and image and extract coordinates for every intersection on the board'''

    # [load_image]
    # Check number of arguments
    # Load the image
    src = cv2.imread(squareFile, cv2.IMREAD_COLOR)
    src_img = np.copy(src)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image: ' + squareFile)
        return -1
    # Show source image
    #cv2.imshow("src", src)
    # [load_image]
    # [gray]
    # Transform source image to gray if it is not already
    if len(src.shape) != 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src
    # Show gray image
    #show_wait_destroy("gray", gray)
    # [gray]
    # [bin]
    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    gray = cv2.bitwise_not(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 15, -2)
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
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    # Show extracted horizontal lines
    #show_wait_destroy("horizontal", horizontal)

    # [horiz]
    # [vert]
    # Specify size on vertical axis
    rows = vertical.shape[0]
    verticalsize = rows // 40
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    
    # Show extracted vertical lines
    #show_wait_destroy("vertical", vertical)

    bwv = cv2.adaptiveThreshold(vertical, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 15, -2)
    bwh = cv2.adaptiveThreshold(horizontal, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 15, -2)
    v_h_and = cv2.bitwise_and(bwv,bwh)

    #show_wait_destroy("and", v_h_and)

    ### finding contours, can use connectedcomponents aswell
    contours,_ = cv2.findContours(v_h_and, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ### converting to bounding boxes from polygon
    contours=[cv2.boundingRect(cnt) for cnt in contours]
    img = np.zeros((src.shape[0],src.shape[1],3), np.uint8)
    ### drawing rectangle for each contour for visualising
    for cnt in contours:
        x,y,w,h=cnt
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),10)
    #show_wait_destroy("and", img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours,_ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ### converting to bounding boxes from polygon
    contours=[cv2.boundingRect(cnt) for cnt in contours]
    img = np.zeros((src.shape[0],src.shape[1],3), np.uint8)
    ### drawing rectangle for each contour for visualising
    transparent_background = np.zeros((720, 720, 4))
    coord_list =[]

    # remove any extra points by searching for points that have too close a neighbor and removing them
    for cnt in contours:
        x,y,w,h=cnt
        #cv2.rectangle(src,(x,y),(x+w,y+h),(0,255,0),2)

        center_coords = (int((x+w/2)), int((y+h/2)))
        coord_list.append(center_coords)
        cv2.circle(transparent_background, center_coords, 5, (255, 255, 255), -1)
    #show_wait_destroy("intersections", transparent_background)

    # remove any extra points by searching for points that have too close a neighbor and removing them
    pared_coords = []
    # sort coordinates by x axis
    coord_list_i = sorted(coord_list, key = lambda x: x[0])
    coord_listy = sorted(coord_list, key = lambda x: x[0])
    rejected_coords = set()
    #print(coord_list)
    for icoord in coord_list_i:
        for jcoord in coord_listy:
            if icoord == jcoord or jcoord in rejected_coords:
                continue
            if abs(icoord[0] - jcoord[0]) < 20:
                if abs(icoord[1] - jcoord[1]) < 20:
                    coord_list_i.remove(jcoord)
                    coord_listy.remove(jcoord)
                    rejected_coords.add(jcoord)
                    break
        pared_coords.append(icoord)
        
    print(len(contours))
    print(len(pared_coords))

    test_image = np.copy(src_img)
    detected_coords = pared_coords

    cv2.namedWindow(winname = "Add missing intersections") 
    
    cv2.setMouseCallback("Add missing intersections", draw_circle) 
    
    while True: 
        cv2.imshow("Add missing intersections", test_image) 
        
        if cv2.waitKey(10) & 0xFF == 27: 
            break


    grid_background = np.zeros((720, 720, 4))
    for coord in pared_coords:
        x,y = coord

        #make a 60px square cocentric with the contour
        #this represents each individual area that will be evaulated
        cv2.rectangle(src,(x-15,y-15),(x+15,y+15),(128,0,128),2)
        cv2.circle(src, coord, 3, (255, 255, 255), -1)

        cv2.circle(grid_background, coord, 5, (255, 255, 255), -1)
    
    #show_wait_destroy("evaluation areas", src)
    

    filename = 'gridfiles/evaluation_grid.png'
    cv2.imwrite(filename, grid_background) 
    
    return pared_coords

def evaluate_board_state(src_file):

    board_size = 19
    evaluation_map = {}
    # open image
    with open('grid_coords.data', 'rb') as filehandle:
        # read the data as binary data stream
        coordinates = pickle.load(filehandle)

    src = cv2.imread(src_file)
    display_img = np.copy(src)
    empty_board = cv2.imread(r'result\flatboard.jpg')
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image: ' + squareFile)
        return -1

    for coord in coordinates:
        x,y = coord
        eval_width = 12
        ymin = y-eval_width if y > eval_width else 0
        ymax = y+eval_width if y < 720 - eval_width else 720
        xmin = x-eval_width if x > eval_width else 0
        xmax = x+10 if x < 720 - eval_width else 720
        crop_img = src[ymin:ymax, xmin:xmax]
        color = evaluate_image(crop_img)
        evaluation_map[coord] = color
        cv2.rectangle(display_img,(x-eval_width,y-eval_width),(x+eval_width,y+eval_width),(128,0,128),2)
        if color == "black":
            cv2.circle(empty_board, coord, 15, (0, 0, 0), -1)
        if color == "white":
            cv2.circle(empty_board, coord, 15, (248,248,255), -1)

    # show_wait_destroy("evaluation_area", display_img)
    show_wait_destroy("calculated state", empty_board)
        

    matrix = map_to_goban_coordinates(coordinates)
    s = [[str(evaluation_map[e]) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))

    board = s
    with open('board_state.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(board, filehandle)

    next_move = save_SGF(board)
    print(next_move)
    x, y = next_move[1]
    color = next_move[0]
    response_coord = matrix[(board_size-1) - x][y]
    if color == "b":
            cv2.circle(empty_board, response_coord, 15, (0, 0, 0), -1)
            cv2.circle(empty_board, response_coord, 9, (255, 255, 255), 2)
    if color == "w":
        cv2.circle(empty_board, response_coord, 15, (255, 255, 255), -1)
        cv2.circle(empty_board, response_coord, 9, (0, 0, 0), 2)
    return (x,y,response_coord, empty_board)

def crop_and_save(src_file, out_path):
    ''' crop a board into images of each individual intersection and save them to disk to create the training dataset'''
    # open image
    with open('grid_coords.data', 'rb') as filehandle:
        # read the data as binary data stream
        grid_coords = pickle.load(filehandle)

    src = cv2.imread(src_file)
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
        #cv2.imwrite(out_path+str(uuid.uuid4())+".jpg", crop_img) 

    for coord in grid_coords:
        x,y = coord
        ymin = y-15 if y > 15 else 0
        ymax = y+15 if y < 585 else 600
        xmin = x-15 if x > 15 else 0
        xmax = x+15 if x < 585 else 600
        # make a 60px square cocentric with the contour
        cv2.rectangle(src,(xmin,ymin),(xmax,ymax),(128,0,128),2)
    show_wait_destroy("intersections", src)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        crop_and_save(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        evaluate_board_state(sys.argv[1])
    else: 
        process_analysis_grid(sys.argv[1:][0])