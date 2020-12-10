from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import cv2
import numpy as np

from gridDetect import process_analysis_grid
SIDE_LENGTH = 570

def image_resize(image, maxLength = 570, inter = cv2.INTER_AREA):
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



class PerspectiveTransform():
    def __init__(self, master):
        self.parent = master
        self.coord = [] 	# x,y coordinate
        self.dot = []
        self.file = '' 	 	#image path
        self.filename ='' 	#image filename
        
        #setting up a tkinter canvas with scrollbars
        self.frame = Frame(self.parent, bd=2, relief=SUNKEN)
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        self.xscroll = Scrollbar(self.frame, orient=HORIZONTAL)
        self.xscroll.grid(row=1, column=0, sticky=E+W)
        self.yscroll = Scrollbar(self.frame)
        self.yscroll.grid(row=0, column=1, sticky=N+S)
        self.canvas = Canvas(self.frame, bd=0, xscrollcommand=self.xscroll.set, yscrollcommand=self.yscroll.set, width=570, height=570)
        self.canvas.grid(row=0, column=0, sticky=N+S+E+W)
        self.xscroll.config(command=self.canvas.xview)
        self.yscroll.config(command=self.canvas.yview)
        self.frame.pack(fill=BOTH,expand=1)
        self.addImage()
        
        #mouseclick event and button
        self.canvas.bind("<Button 1>",self.insertCoords)
        self.canvas.bind("<Button 3>",self.removeCoords)
        self.ctrPanel = Frame(self.frame)
        self.ctrPanel.grid(row = 0, column = 2, columnspan = 2, sticky = N+E)
        self.addImgBtn = Button(self.ctrPanel, text="Browse", command=self.addImage)
        self.addImgBtn.grid(row=0,column=2, pady = 5, sticky =NE)
        self.saveBtn = Button(self.ctrPanel, text="Save", command=self.saveImage)
        self.saveBtn.grid(row=1,column=2, pady = 5, sticky =NE)
        self.undoBtn = Button(self.ctrPanel, text="Undo", command=self.undo)
        self.undoBtn.grid(row=2,column=2, pady = 5, sticky =NE)
        self.loadBtn = Button(self.ctrPanel, text="Load grid", command=self.load_analysis_grid)
        self.loadBtn.grid(row=3,column=2, pady = 5, sticky =NE)
    
    #adding the image
    def addImage(self):
        self.coord = []
        self.file = askopenfilename(parent=self.parent, initialdir="image/",title='Choose an image.')
        self.filename = self.file.split('/')[-1]
        self.filename = self.filename.rstrip('.jpg')
        img = image_resize(cv2.imread(self.file), maxLength = 570, inter = cv2.INTER_AREA)
        self.cv_img = img
        self.last_img = img
        b,g,r = cv2.split(img)
        img = cv2.merge((r,g,b))
        self.img = ImageTk.PhotoImage(image = Image.fromarray(img))
        self.starting_image = self.img
        
        self.canvas.create_image(0,0,image=self.img,anchor="nw")
        self.canvas.config(scrollregion=self.canvas.bbox(ALL), width=570, height=570)
        
    def undo(self):
        self.cv_img = self.last_img
        img = self.cv_img
        self.canvas.delete("all")
        b,g,r = cv2.split(img)
        img = cv2.merge((r,g,b))
        self.img = ImageTk.PhotoImage(image = Image.fromarray(img))
        self.canvas.create_image(0,0,image=self.img,anchor="nw")
        self.coord = []


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
        frame = self.cv_img #image_resize(cv2.imread(self.file), maxLength = 570, inter = cv2.INTER_AREA)
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
     
        print(self.coord)
        pts1 = np.float32(self.coord)    
        pts2 = np.float32([[0, 0], [maxSide-1, 0], [0, maxSide-1], [maxSide-1, maxSide-1]])
        self.pts1 = pts1
        self.pts2 = pts2
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        self.result_cv = cv2.warpPerspective(frame, matrix, (maxSide,maxSide))
         
        #cv2.imshow("Frame", frame_circle)
        #cv2.imshow("Perspective transformation", result_cv)
        
        result_rgb = cv2.cvtColor(self.result_cv, cv2.COLOR_BGR2RGB)
        self.result = ImageTk.PhotoImage(image = Image.fromarray(result_rgb))
        self.cv_img = self.result_cv
        
    def saveImage(self):
        filename = "result/"+self.filename+"_res.jpg"
        cv2.imwrite(filename, self.result_cv)
        print(self.filename+" is saved!")
        process_analysis_grid(filename)

    def load_analysis_grid(self):
        filename = "gridfiles/evaluation_grid.png"
        gridImage = image_resize(cv2.imread(filename), maxLength = 570, inter = cv2.INTER_AREA)
        frame_circle = gridImage.copy()

        cv2.circle(frame_circle, tuple(self.coord[0]), 5, (0, 0, 255), -1)
        cv2.circle(frame_circle, tuple(self.coord[1]), 5, (0, 0, 255), -1)
        cv2.circle(frame_circle, tuple(self.coord[2]), 5, (0, 0, 255), -1)
        cv2.circle(frame_circle, tuple(self.coord[3]), 5, (0, 0, 255), -1)
     
        pts1 = np.float32(self.coord)    
        pts2 = np.float32([[0, 0], [SIDE_LENGTH-1, 0], [0, SIDE_LENGTH-1], [SIDE_LENGTH-1, SIDE_LENGTH-1]])
        matrix = cv2.getPerspectiveTransform(self.pts2, self.pts1)
        grid_result = cv2.warpPerspective(gridImage, matrix, (SIDE_LENGTH,SIDE_LENGTH))
        src = image_resize(cv2.imread(self.file), maxLength = 570, inter = cv2.INTER_AREA)
        src = cv2.copyMakeBorder( src, 0, grid_result.shape[0]-src.shape[0], 0, grid_result.shape[1]-src.shape[1], cv2.BORDER_CONSTANT)
        cv2.imshow("Perspective transformation", cv2.add(src,grid_result))
        cv2.imshow("Grid", grid_result)

        



#---------------------------------
if __name__ == '__main__':
    root = Tk()
    #root.geometry("500x500")
    transformer = PerspectiveTransform(root)
    root.mainloop()