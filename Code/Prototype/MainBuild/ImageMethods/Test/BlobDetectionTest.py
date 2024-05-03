"""
BlobDetectionTest
Author: Kacper Dziedzic ktd1
Version: 1.1

Code to test out Blob Detection for image preprocessing
"""

import cv2
import numpy as np

# Reading an image
img = cv2.imread("C:/Users/Katch/Desktop/grain/broken/broken over 1_5mm001.tif")
img = cv2.resize(img, (1000, 1000))

# The kernel to be used for dilation purpose
kernel = np.ones((1, 1), np.uint8)

# converting the image to HSV format
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# defining the lower and upper values of HSV,
# this will detect yellow colour
Lower_hsv = np.array([0, 55, 45])
Upper_hsv = np.array([255, 255, 255])

# creating the mask by eroding,morphing,
# dilating process
Mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv)
Mask = cv2.erode(Mask, kernel, iterations=1)
Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel)
Mask = cv2.dilate(Mask, kernel, iterations=1)

# Inverting the mask by
# performing bitwise-not operation
Mask = cv2.bitwise_not(Mask)


params = cv2.SimpleBlobDetector_Params()

# Set Area filtering parameters
params.filterByArea = True
params.minArea = 25

# Set Circularity filtering parameters
params.filterByCircularity = True
params.minCircularity = 0.1

# Set Convexity filtering parameters
params.filterByConvexity = True
params.minConvexity = 0.1

# Set inertia filtering parameters
params.filterByInertia = True
params.minInertiaRatio = 0.01

detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(Mask)

# Draw blobs on our image as red circles
blank = np.zeros((1, 1))
blobs = cv2.drawKeypoints(Mask, keypoints, blank, (0, 0, 255),
                          cv2.DRAW_MATCHES_FLAGS_DEFAULT)

number_of_blobs = len(keypoints)
text = "Number of Circular Blobs: " + str(len(keypoints))
cv2.putText(blobs, text, (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

# Show blobs
cv2.imshow("Output", blobs)
cv2.waitKey(0)