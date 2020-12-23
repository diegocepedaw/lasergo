import cv2 
path ="average_images\white_average.jpg"
  
# reading the image in grayscale mode 
gray = cv2.imread(path, 0) 
  
# threshold 
th, threshed = cv2.threshold(gray, 100, 255,  
          cv2.THRESH_BINARY|cv2.THRESH_OTSU) 
  
# findcontours 
cnts = cv2.findContours(threshed, cv2.RETR_LIST,  
                    cv2.CHAIN_APPROX_SIMPLE)[-2] 
  
# filter by area 
s1 = 35
s2 = 100
xcnts = [] 
  
for cnt in cnts: 
    if s1<cv2.contourArea(cnt) <s2:
        x,y,w,h = cv2.boundingRect(cnt)
        center_coords = (int((x+w/2)), int((y+h/2)))
        cv2.circle(gray, center_coords, 10, (255, 0, 0), 2) 
        xcnts.append(cnt) 
cv2.imshow('image',gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
# printing output 
print("\nDots number: {}".format(len(xcnts))) 