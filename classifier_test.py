#here is code to check the output
import cv2
import numpy as np
import webcolors




#######################
#try straightening again with te top 4 detected corners
#######################
# def rgb_to_hex(rgb):
#     return '%02x%02x%02x' % rgb


face_cascade=cv2.CascadeClassifier(r"C:\Users\diego\Documents\lasergo\classifiers\black_stone\cascade.xml")###path of cascade file
## following is an test image u can take any image from the p folder in the temp folder and paste address of it on below line 
img= cv2.imread(r"C:\Users\diego\Documents\lasergo\training_images\black_stone\0a1ba965-a068-43a2-8fae-539c53c499eb.jpg")###path of image file which we want to detect

colors, count = np.unique(img.reshape(-1,img.shape[-1]), axis=0, return_counts=True)
rgb = colors[count.argmax()]
rgb = (rgb[0],rgb[1],rgb[2])
print(rgb)
color_img = np.zeros((30,30,3), np.uint8)
color_img[:] = rgb
cv2.imshow("color", color_img)
hex_rgb = '#%02x%02x%02x' % rgb
print(hex_rgb)
#print(webcolors.rgb_to_name((rgb[0],rgb[1],rgb[2]), spec="html4"))
if((rgb[0] < 50) and (rgb[1] < 50) and (rgb[2] < 50)):
    print("black")
elif((rgb[0] > 200) and (rgb[1] > 200) and (rgb[2] > 200)):
    print("White")
else:
    print("board")
cv2.waitKey(0)

img= cv2.imread(r"C:\Users\diego\Documents\lasergo\training_images\white stone\0a61eb05-c266-4def-abc1-f5cd2c4f0d84.jpg")###path of image file which we want to detect

colors, count = np.unique(img.reshape(-1,img.shape[-1]), axis=0, return_counts=True)
rgb = colors[count.argmax()]

color_img = np.zeros((30,30,3), np.uint8)
color_img[:] = (rgb[0],rgb[1],rgb[2])
cv2.imshow("color", color_img)
print(rgb)

if((rgb[0] < 50) and (rgb[1] < 50) and (rgb[2] < 50)):
    print("black")
elif((rgb[0] > 200) and (rgb[1] > 200) and (rgb[2] > 200)):
    print("White")
else:
    print("board")

cv2.waitKey(0)


img= cv2.imread(r"C:\Users\diego\Documents\lasergo\training_images\empty_point\0d7c76fd-cbdf-4d56-a74e-629d496b4905.jpg")###path of image file which we want to detect

colors, count = np.unique(img.reshape(-1,img.shape[-1]), axis=0, return_counts=True)
rgb = colors[count.argmax()]

color_img = np.zeros((30,30,3), np.uint8)
color_img[:] = (rgb[0],rgb[1],rgb[2])
cv2.imshow("color", color_img)
print(rgb)

if((rgb[0] < 50) and (rgb[1] < 50) and (rgb[2] < 50)):
    print("black")
elif((rgb[0] > 200) and (rgb[1] > 200) and (rgb[2] > 200)):
    print("White")
else:
    print("board")

cv2.waitKey(0)

# faces=face_cascade.detectMultiScale(img)
# print(len(faces))
# ##if not getting good result try to train new cascade.xml file again deleting other file expect p and n in temp folder
# for(x,y,w,h) in faces:
#     resized=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),-1)
# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()