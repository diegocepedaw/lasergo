import cv2
import os
   
      
# test image 
image = cv2.imread(r'..\training_images\black_stone\0a1ba965-a068-43a2-8fae-539c53c499eb.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
histogram = cv2.calcHist([gray_image], [0],  
                         None, [256], [0, 256]) 

# black stone image 
image = cv2.imread(r'..\average_images\black_average.jpg') 
gray_image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
histogram1 = cv2.calcHist([gray_image1], [0],  
                          None, [256], [0, 256]) 
   

# white stone image 
image = cv2.imread(r'..\average_images\white_average.jpg') 
gray_image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
histogram2 = cv2.calcHist([gray_image2], [0],  
                          None, [256], [0, 256]) 

# empty point image 
image = cv2.imread(r'..\average_images\point_average.jpg') 
gray_image3 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
histogram3 = cv2.calcHist([gray_image3], [0],  
                          None, [256], [0, 256])
   

test_images = 1000

# test black images
c1, c2, c3 = 0, 0, 0
iterations = 0
mistakes = 0
f = r'..\training_images\black_stone'
for file in os.listdir(f):
    f_img = f+"/"+file
    image = cv2.imread(f_img)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    histogram = cv2.calcHist([gray_image], [0],  
                            None, [256], [0, 256]) 

    # Euclidean Distace between data1 and test 
    i = 0
    while i<len(histogram) and i<len(histogram1): 
        c1+=(histogram[i]-histogram1[i])**2
        i+= 1
    c1 = c1**(1 / 2) 
    
    # Euclidean Distace between data2 and test 
    i = 0
    while i<len(histogram) and i<len(histogram2): 
        c2+=(histogram[i]-histogram2[i])**2
        i+= 1
    c2 = c2**(1 / 2) 

    # Euclidean Distace between data2 and test 
    i = 0
    while i<len(histogram) and i<len(histogram3): 
        c3+=(histogram[i]-histogram3[i])**2
        i+= 1
    c3 = c3**(1 / 2) 

    if c1 > c2 or c1 > c3:
        mistakes += 1
    
    #img = Image.open(f_img)
    if iterations > test_images:
        print("Black test:")
        print("correctly labeled: %s", str(iterations-mistakes))
        print("incorrectly labeled: %s", str(mistakes))
        break
    iterations += 1 


# test white images
c1, c2, c3 = 0, 0, 0
iterations = 0
mistakes = 0
f = r'..\training_images\white_stone'
for file in os.listdir(f):
    f_img = f+"/"+file
    image = cv2.imread(f_img)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    histogram = cv2.calcHist([gray_image], [0],  
                            None, [256], [0, 256]) 

    # Euclidean Distace between data1 and test 
    i = 0
    while i<len(histogram) and i<len(histogram1): 
        c1+=(histogram[i]-histogram1[i])**2
        i+= 1
    c1 = c1**(1 / 2) 
    
    # Euclidean Distace between data2 and test 
    i = 0
    while i<len(histogram) and i<len(histogram2): 
        c2+=(histogram[i]-histogram2[i])**2
        i+= 1
    c2 = c2**(1 / 2) 

    # Euclidean Distace between data2 and test 
    i = 0
    while i<len(histogram) and i<len(histogram3): 
        c3+=(histogram[i]-histogram3[i])**2
        i+= 1
    c3 = c3**(1 / 2) 

    if c2 > c1 or c2 > c3:
        mistakes += 1
    
    #img = Image.open(f_img)
    if iterations > test_images:
        print("White test:")
        print("correctly labeled: %s", str(iterations-mistakes))
        print("incorrectly labeled: %s", str(mistakes))
        break
    iterations += 1 
   
