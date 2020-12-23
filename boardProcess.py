from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import cv2
import numpy as np
import pickle
import uuid

from webcam_util import start_capture, capture_image
from gridDetect import process_analysis_grid, show_wait_destroy, evaluate_board_state
from laser_utils import target_laser
SIDE_LENGTH = 720

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

def crop_and_save(src_file, out_path):
    ''' crop a board into images of each individual intersection and save them to disk to create the training dataset'''
    # open image
    with open('grid_coords.data', 'rb') as filehandle:
        # read the data as binary data stream
        grid_coords = pickle.load(filehandle)

    with open('corner_original_coords.data', 'rb') as filehandle:
        # read the data as binary data stream
        pts1 = pickle.load(filehandle)
    
    with open('corner_transformed_coords.data', 'rb') as filehandle:
        # read the data as binary data stream
        pts2 = pickle.load(filehandle)

    src = cv2.imread(src_file)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image: ' + squareFile)
        return -1

    src = image_resize(src, maxLength = 720, inter = cv2.INTER_AREA)
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    src = cv2.warpPerspective(src, matrix, (720,720))

    for coord in grid_coords:
        x,y = coord
        ymin = y-15 if y > 15 else 0
        ymax = y+15 if y < 585 else 600
        xmin = x-15 if x > 15 else 0
        xmax = x+15 if x < 585 else 600
        crop_img = src[ymin:ymax, xmin:xmax]
        #show_wait_destroy("crop", crop_img)
        cv2.imwrite(out_path+str(uuid.uuid4())+".jpg", crop_img) 

    for coord in grid_coords:
        x,y = coord
        ymin = y-15 if y > 15 else 0
        ymax = y+15 if y < 585 else 600
        xmin = x-15 if x > 15 else 0
        xmax = x+15 if x < 585 else 600
        # make a 60px square cocentric with the contour
        cv2.rectangle(src,(xmin,ymin),(xmax,ymax),(128,0,128),2)
    show_wait_destroy("intersections", src)



class PerspectiveTransform():
    def __init__(self, master):
        self.parent = master
        self.coord = [] 	# x,y coordinate
        self.dot = []
        self.file = '' 	 	#image path
        self.filename ='' 	#image filename

        # set up capture camera
        self.camera = cv2.VideoCapture(0)
        
        #setting up a tkinter canvas with scrollbars
        self.frame = Frame(self.parent, bd=2, relief=SUNKEN)
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        self.xscroll = Scrollbar(self.frame, orient=HORIZONTAL)
        self.xscroll.grid(row=1, column=0, sticky=E+W)
        self.yscroll = Scrollbar(self.frame)
        self.yscroll.grid(row=0, column=1, sticky=N+S)
        self.canvas = Canvas(self.frame, bd=0, xscrollcommand=self.xscroll.set, yscrollcommand=self.yscroll.set, width=720, height=720)
        self.canvas.grid(row=0, column=0, sticky=N+S+E+W)
        self.xscroll.config(command=self.canvas.xview)
        self.yscroll.config(command=self.canvas.yview)
        self.frame.pack(fill=BOTH,expand=1)
        self.addImage()
        
        #mouseclick event and button
        self.canvas.bind("<Button 1>",self.insertCoords)
        self.canvas.bind("<Button 3>",self.removeCoords)
        self.canvas.bind_all("<space>", self.captureFrame)
        self.ctrPanel = Frame(self.frame)
        self.ctrPanel.grid(row = 0, column = 2, columnspan = 2, sticky = N+E)
        self.addImgBtn = Button(self.ctrPanel, text="Browse", command=self.addImage)
        self.addImgBtn.grid(row=0,column=2, pady = 5, sticky =NE)
        self.initBtn = Button(self.ctrPanel, text="Initialize board", command=self.initImage)
        self.initBtn.grid(row=1,column=2, pady = 5, sticky =NE)
        self.undoBtn = Button(self.ctrPanel, text="Undo", command=self.undo)
        self.undoBtn.grid(row=2,column=2, pady = 5, sticky =NE)
        self.loadBtn = Button(self.ctrPanel, text="Load grid", command=self.load_analysis_grid)
        self.loadBtn.grid(row=3,column=2, pady = 5, sticky =NE)
        self.loadBtn = Button(self.ctrPanel, text="slice board", command=self.slice)
        self.loadBtn.grid(row=4,column=2, pady = 5, sticky =NE)
        self.saveBtn = Button(self.ctrPanel, text="Save board", command=self.saveBoard)
        self.saveBtn.grid(row=5,column=2, pady = 5, sticky =NE)
        self.captureBtn = Button(self.ctrPanel, text="Capture frame", command=self.captureFrame)
        self.captureBtn.grid(row=6,column=2, pady = 5, sticky =NE)

    #adding the image
    def addImage(self):
        self.coord = []
        # self.file = askopenfilename(parent=self.parent, initialdir="image/",title='Choose an image.')
        # self.filename = self.file.split('/')[-1]
        # self.filename = self.filename.rstrip('.jpg')
        # img = image_resize(cv2.imread(self.file), maxLength = 720, inter = cv2.INTER_AREA)

        img = capture_image(self.camera)
        self.full_res_orig = np.copy(img)

        img = image_resize(img, maxLength = 720, inter = cv2.INTER_AREA)
        self.original_image = np.copy(img)
        self.cv_img = img
        self.last_img = img
        b,g,r = cv2.split(img)
        img = cv2.merge((r,g,b))
        self.img = ImageTk.PhotoImage(image = Image.fromarray(img))
        self.starting_image = self.img
        
        self.canvas.create_image(0,0,image=self.img,anchor="nw")
        self.canvas.config(scrollregion=self.canvas.bbox(ALL), width=720, height=720)
        
    def undo(self):
        self.cv_img = self.last_img
        img = self.cv_img
        self.canvas.delete("all")
        b,g,r = cv2.split(img)
        img = cv2.merge((r,g,b))
        self.img = ImageTk.PhotoImage(image = Image.fromarray(img))
        self.canvas.create_image(0,0,image=self.img,anchor="nw")
        self.coord = []

    def reverse_coord_transform(self, response_img):

        matrix = cv2.getPerspectiveTransform(self.pts2, self.pts1)
        result = cv2.warpPerspective(response_img, matrix, (SIDE_LENGTH,SIDE_LENGTH))
        src = np.copy(self.original_image)
        src = cv2.copyMakeBorder( src, 0, result.shape[0]-src.shape[0], 0, result.shape[1]-src.shape[1], cv2.BORDER_CONSTANT)

        thresh = cv2.threshold(result[:, :, 1], 25, 255, cv2.THRESH_BINARY)[1]
        lel = cv2.findNonZero(thresh)
        x = 0
        y = 0
        if lel is not None:
            for element in lel:
                y += element[0][0]
                x += element[0][1]
            x = x / len(lel)
            y = y / len(lel)

        response_coord = (int(y),int(x))
        orig_img = np.copy(self.original_image)
        cv2.circle(orig_img, response_coord, 10, (255, 255, 255), -1)
        show_wait_destroy("placement", orig_img)
        
        print(response_coord)
        target_laser(response_coord)


    def captureFrame(self, event=None):
        maxSide = 720
        img = capture_image(self.camera)
        matrix = cv2.getPerspectiveTransform(self.pts1, self.pts2)
        img = image_resize(img, maxLength = 720, inter = cv2.INTER_AREA)
        img = cv2.warpPerspective(img, matrix, (maxSide,maxSide))
        show_wait_destroy("image", img)
        img_name = "captured_images/web_capture.png"
        cv2.imwrite(img_name, img)
        print("{} written!".format(img_name))
        response_point_img = evaluate_board_state("captured_images/web_capture.png")
        response_coord = self.reverse_coord_transform(response_point_img)
    
    #Save coord according to mouse left click
    def insertCoords(self, event):
        if (len(self.coord) == 4):
            self.coord = []

        #outputting x and y coords to console
        self.coord.append([event.x, event.y])
        r=3
        self.dot.append(self.canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill="#ff0000"))         #print circle
        if (len(self.coord) == 4):
            self.Transformer()
            self.canvas.delete("all")
            self.canvas.create_image(0,0,image=self.result,anchor="nw")
            self.canvas.image = self.result
            #self.create_grid()
    
    #remove last inserted coord using mouse right click
    def removeCoords(self, event=None):
        del self.coord[-1]
        self.canvas.delete(self.dot[-1])
        del self.dot[-1]
    
    def create_grid(self, event=None):
        w = self.canvas.winfo_width() # Get current width of canvas
        h = self.canvas.winfo_height() # Get current height of canvas
        self.canvas.delete('grid_line') # Will only remove the grid_line

        # Creates all vertical lines at intevals of 30
        for i in range(0, w, 30):
            self.canvas.create_line([(i, 0), (i, h)], tag='grid_line')

        # Creates all horizontal lines at intevals of 30
        for i in range(0, h, 30):
            self.canvas.create_line([(0, i), (w, i)], tag='grid_line')

    
    def Transformer(self):   
        frame = self.cv_img #image_resize(cv2.imread(self.file), maxLength = 720, inter = cv2.INTER_AREA)
        self.last_img = frame
        frame_circle = frame.copy()
        #points = [[480,90],[680,90],[0,435],[960,435]]
        cv2.circle(frame_circle, tuple(self.coord[0]), 5, (0, 0, 255), -1)
        cv2.circle(frame_circle, tuple(self.coord[1]), 5, (0, 0, 255), -1)
        cv2.circle(frame_circle, tuple(self.coord[2]), 5, (0, 0, 255), -1)
        cv2.circle(frame_circle, tuple(self.coord[3]), 5, (0, 0, 255), -1)
        
        widthA = np.sqrt(((self.coord[3][0] - self.coord[2][0]) ** 2) + ((self.coord[3][1] - self.coord[2][1]) ** 2))
        widthB = np.sqrt(((self.coord[1][0] - self.coord[0][0]) ** 2) + ((self.coord[1][1] - self.coord[0][1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
         
        heightA = np.sqrt(((self.coord[1][0] - self.coord[3][0]) ** 2) + ((self.coord[1][1] - self.coord[3][1]) ** 2))
        heightB = np.sqrt(((self.coord[0][0] - self.coord[2][0]) ** 2) + ((self.coord[0][1] - self.coord[2][1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        maxSide = SIDE_LENGTH #max(maxHeight, maxWidth)
     
        #print(self.coord)
        pts1 = np.float32(self.coord)    
        pts2 = np.float32([[0, 0], [maxSide-1, 0], [0, maxSide-1], [maxSide-1, maxSide-1]])
        self.pts1 = pts1
        self.pts2 = pts2
        matrix = cv2.getPerspectiveTransform(self.pts1, self.pts2)
        self.result_cv = cv2.warpPerspective(frame, matrix, (maxSide,maxSide))
         
        #cv2.imshow("Frame", frame_circle)
        #cv2.imshow("Perspective transformation", result_cv)
        
        result_rgb = cv2.cvtColor(self.result_cv, cv2.COLOR_BGR2RGB)
        self.result = ImageTk.PhotoImage(image = Image.fromarray(result_rgb))
        self.cv_img = self.result_cv
        
    def initImage(self):
        filename = r"result\flatboard.jpg"
        cv2.imwrite(filename, self.result_cv)
        print(self.filename+" is saved!")
        self.grid_coords = process_analysis_grid(filename)
        with open('grid_coords.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(self.grid_coords, filehandle)
        with open('corner_original_coords.data', 'wb') as filehandle:
            pickle.dump(self.pts1, filehandle)
        with open('corner_transformed_coords.data', 'wb') as filehandle:
            pickle.dump(self.pts2, filehandle)

    def saveBoard(self):
        maxSide = 720
        src = cv2.imread(r'test_images\WIN_20201220_22_30_35_Pro.jpg')
        src = image_resize(src)
        matrix = cv2.getPerspectiveTransform(self.pts1, self.pts2)
        self.result_cv = cv2.warpPerspective(src, matrix, (maxSide,maxSide))

        show_wait_destroy("intersections", src)
        filename = "result/boardstate.jpg"
        cv2.imwrite(filename, self.result_cv)
        print(self.filename+" is saved!")
       
    def load_analysis_grid(self):
        filename = "gridfiles\evaluation_grid.png"
        gridImage = image_resize(cv2.imread(filename), maxLength = 720, inter = cv2.INTER_AREA)


        pts1 = np.float32(self.coord)    
        pts2 = np.float32([[0, 0], [SIDE_LENGTH-1, 0], [0, SIDE_LENGTH-1], [SIDE_LENGTH-1, SIDE_LENGTH-1]])
        matrix = cv2.getPerspectiveTransform(self.pts2, self.pts1)
        result = cv2.warpPerspective(gridImage, matrix, (SIDE_LENGTH,SIDE_LENGTH))
        src = np.copy(self.original_image)
        src = cv2.copyMakeBorder( src, 0, result.shape[0]-src.shape[0], 0, result.shape[1]-src.shape[1], cv2.BORDER_CONSTANT)
        cv2.imshow("Perspective transformation", cv2.add(src,result))
        cv2.imshow("Grid", result)

        
    def slice(self):
        crop_and_save("result/flatboard.jpg", "training_images\empty_point" )






#---------------------------------
if __name__ == '__main__':
    if len(sys.argv) > 2:
        crop_and_save(sys.argv[1], sys.argv[2])
    else: 
        root = Tk()
        transformer = PerspectiveTransform(root)
        root.mainloop()