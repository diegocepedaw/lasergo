# import the necessary packages
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import cv2
import os


index = {}
images = {}

# loop over the image paths
iterations = 0
mistakes = 0

filename = r'average_images\black_average.jpg'
image = cv2.imread(filename)
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
images["board"] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# extract a 3D RGB color histogram from the image,
# using 8 bins per channel, normalize, and update
# the index
hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
    [0, 256, 0, 256, 0, 256])
hist = cv2.normalize(hist, hist).flatten()
index["board"] = hist



# METHOD #1: UTILIZING OPENCV
# initialize OpenCV methods for histogram comparison
OPENCV_METHODS = (
	("Correlation", cv2.HISTCMP_CORREL),
	("Chi-Squared", cv2.HISTCMP_CHISQR),
	("Intersection", cv2.HISTCMP_INTERSECT),
	("Hellinger", cv2.HISTCMP_BHATTACHARYYA))



test_images = 10


# loop over the comparison methods
for (methodName, method) in OPENCV_METHODS:
	# initialize the results dictionary and the sort
	# direction
    results = {}
    reverse = False
    # if we are using the correlation or intersection
    # method, then sort the results in reverse order
    if methodName in ("Correlation", "Intersection"):
        reverse = True
    print("\n\n")
    print(methodName)

    iterations = 0
    mistakes = 0
    f = r'training_images\white_stone'
    for file in os.listdir(f):
        f_img = f+"/"+file
        image = cv2.imread(f_img)
        test_hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
            [0, 256, 0, 256, 0, 256])
        test_hist = cv2.normalize(test_hist, test_hist).flatten()

        # loop over the index
        for (k, hist) in index.items():
            # compute the distance between the two histograms
            # using the method and update the results dictionary
            d = cv2.compareHist(test_hist, hist, method)
            results[k] = d
        
        print(sorted([(v, k) for (k, v) in results.items()], reverse = reverse))

        if iterations > test_images:
            break
        iterations += 1 
	# sort the results