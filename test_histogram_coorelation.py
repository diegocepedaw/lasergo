# import the necessary packages
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import cv2
import os
import pickle
from image_processing_utils import apply_processing

index = {}
images = {}

def evaluate_image(image, answer=None):
    if len(index) == 0:
        initialize_references()
    # apply Bilateral Filtering
    image = cv2.bilateralFilter(image,9,75,75)
    results = {}
    test_hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
        [0, 256, 0, 256, 0, 256])
    test_hist = cv2.normalize(test_hist, test_hist).flatten()

    # loop over the index
    for (k, hist) in index.items():
        # compute the distance between the two histograms
        # using the method and update the results dictionary
        d = cv2.compareHist(test_hist, hist, cv2.HISTCMP_CORREL)
        results[k] = d
    
    correlations = sorted([(v, k) for (k, v) in results.items()], reverse = True)
    # maximum correlation score to be considered board
    coorelation_threshold = 0.2
    if correlations[0][0] > coorelation_threshold:
        evaluated_type = correlations[0][1]
    else:
        evaluated_type = "board"
    if answer is not None:
        if answer != evaluated_type:
            data = np.reshape(image, (-1,3))
            print(data.shape)
            data = np.float32(data)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            flags = cv2.KMEANS_RANDOM_CENTERS
            compactness,labels,centers = cv2.kmeans(data,1,None,criteria,10,flags)

            print('Dominant color is: bgr({})'.format(centers[0].astype(np.int32)))
            width = 50
            height = 50
            rgb = centers[0].astype(np.int32)
            dominant_color = np.zeros((height,width,3), np.uint8)
            dominant_color[:,0:width//2] = (rgb[0],rgb[1],rgb[2])      # (B, G, R)
            dominant_color[:,width//2:width] = (rgb[0],rgb[1],rgb[2])
            print(correlations)
            cv2.imshow("incorrect answer", image)
            cv2.imshow("dominant color", dominant_color)
            cv2.waitKey(0)

    return evaluated_type

def initialize_references():
    ''' references are generated from the average of a few thousand collected sample images'''
    # initialize reference histograms
    
    filename = r'average_images\black_average.jpg'
    image = cv2.imread(filename)
    image = apply_processing(image)
    # applly blur
    image = cv2.bilateralFilter(image,9,75,75)
    images["black"] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # extract a 3D RGB color histogram from the image,
    # using 8 bins per channel, normalize, and update
    # the index
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    index["black"] = hist

    filename = r'average_images\white_average.jpg'
    image = cv2.imread(filename)
    image = apply_processing(image)
    # applly blur
    image = cv2.bilateralFilter(image,9,75,75)
    images["white"] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # extract a 3D RGB color histogram from the image,
    # using 8 bins per channel, normalize, and update
    # the index
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    index["white"] = hist

    filename = r'average_images\point_average.jpg'
    image = cv2.imread(filename)
    image = apply_processing(image)
    # applly blur
    image = cv2.bilateralFilter(image,9,75,75)
    images["board"] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # extract a 3D RGB color histogram from the image,
    # using 8 bins per channel, normalize, and update
    # the index
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    index["board"] = hist

def test_directories():
    initialize_references()
    evaluate_images = 1000   
    iterations = 0
    mistakes = 0
    f = r'training_images\black_stone'
    color = "black"
    for file in os.listdir(f):
        filename = f+"/"+file
        image = cv2.imread(filename)
        if evaluate_image(image, color) != color:
            mistakes += 1
        #img = Image.open(f_img)
        if iterations > evaluate_images:
            print("%s test:" % color)
            print("correctly labeled: %s" % str(iterations-mistakes))
            print("incorrectly labeled: %s" %  str(mistakes))
            break
        iterations += 1

    iterations = 0
    mistakes = 0
    f = r'training_images\white_stone'
    color = "white"
    for file in os.listdir(f):
        filename = f+"/"+file
        image = cv2.imread(filename)
        if evaluate_image(image, color) != color:
            mistakes += 1
        #img = Image.open(f_img)
        if iterations > evaluate_images:
            print("%s test:" % color)
            print("correctly labeled: %s" % str(iterations-mistakes))
            print("incorrectly labeled: %s" %  str(mistakes))
            break
        iterations += 1

    iterations = 0
    mistakes = 0
    f = r'training_images\empty_point'
    color = "board"
    for file in os.listdir(f):
        filename = f+"/"+file
        image = cv2.imread(filename)
        if evaluate_image(image, color) != color:
            mistakes += 1
        #img = Image.open(f_img)
        if iterations > evaluate_images:
            print("%s test:" % color)
            print("correctly labeled: %s" % str(iterations-mistakes))
            print("incorrectly labeled: %s" %  str(mistakes))
            break
        iterations += 1 

    



if __name__ == "__main__":
    test_directories()