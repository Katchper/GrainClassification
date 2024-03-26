import copy

import cv2
import numpy as np

# Read the image
original_image = cv2.imread("C:/Users/Katch/Desktop/grain/broken/broken over 1_5mm001.tif", cv2.IMREAD_GRAYSCALE)
cropped_image = original_image[1255:3400, 160:2350]
grayCopy = copy.deepcopy(cropped_image)

colour_image = cv2.imread("C:/Users/Katch/Desktop/grain/broken/broken over 1_5mm001.tif")
cropped_image2 = colour_image[1255:3400, 160:2350]
newImg = copy.deepcopy(cropped_image2)
# Thresholding
# Thresholding (invert the threshold to detect bright objects on black background)
_, thresh = cv2.threshold(grayCopy, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Noise removal (optional)
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
_, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

# Apply watershed
markers = cv2.watershed(newImg, markers)
newImg[markers == -1] = [0, 0, 255]  # Mark boundaries in red

cv2.imshow('Separated Objects', newImg)
cv2.waitKey(0)
cv2.destroyAllWindows()