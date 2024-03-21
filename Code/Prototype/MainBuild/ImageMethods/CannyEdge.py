import time

import cv2
import numpy as np
from matplotlib import pyplot as plt

from Code.Prototype.MainBuild.FileMethods.arffBuilder import writeLineToARFF
thresh1 = 0
thresh2 = 0
contrast = 0
brightness = 0

def update_contrast(contrast1):
    global contrast
    contrast = contrast1
    updateScreen2()
def update_brightness(brightness1):
    global brightness
    brightness = brightness1
    updateScreen2()

def update_threshold1(threshold1):
    global thresh1
    thresh1 = threshold1
    updateScreen()
def update_threshold2(threshold2):
    global thresh2
    thresh2 = threshold2
    updateScreen()
def updateScreen():
    cannyEdge = cv2.Canny(cropped_image2, thresh1, thresh2)
    cv2.imshow('CannyEdgeDetection', cannyEdge)
    imgTest = cv2.subtract(threshold1, cannyEdge)
    cv2.imshow("Contour", imgTest)

def updateScreen2():

    brightnesstemp = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
    contrasttemp = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
    alpha12 = (max - brightnesstemp) / 255
    gamma12 = brightnesstemp
    caltest = cv2.addWeighted(out, alpha12, out, 0, gamma12)

    Alpha22 = float(131 * (contrasttemp + 127)) / (127 * (131 - contrasttemp))
    Gamma22 = 127 * (1 - Alpha22)
    caltest = cv2.addWeighted(caltest, Alpha22, caltest, 0, Gamma22)
    cv2.imshow("contrastbrightness", caltest)

original_image = cv2.imread("C:/Users/Katch/Desktop/grain/wholegrain/21QC_002.tif", cv2.IMREAD_GRAYSCALE)
cropped_image = original_image[2000:2800, 160:2350]

# colour image
colour_image = cv2.imread("C:/Users/Katch/Desktop/grain/wholegrain/21QC_002.tif")
cropped_image2 = colour_image[2000:2800, 160:2350]
newImg = cropped_image2

out = cv2.addWeighted(cropped_image, 1, cropped_image, 0, 0)

cannyEdge = cv2.Canny(cropped_image2, 100, 200)

"""
plt.subplot(121), plt.imshow(cropped_image2, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()"""
# contrast + brightness changes
max = 255
brightness = int((325 - 0) * (255 - (-255)) / (510 - 0) + (-255))
contrast = int((254 - 0) * (127 - (-127)) / (254 - 0) + (-127))
alpha1 = (max - brightness) / 255
gamma1 = brightness
cal = cv2.addWeighted(out, alpha1, out, 0, gamma1)

Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
Gamma = 127 * (1 - Alpha)
cal = cv2.addWeighted(cal, Alpha, cal, 0, Gamma)

cv2.namedWindow("contrastbrightness", cv2.WINDOW_NORMAL)
cv2.imshow("contrastbrightness", cal)

# apply a threshold
# ret3, th4 = cv2.threshold(cal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

ret, threshold1 = cv2.threshold(cal, 10, 255, cv2.THRESH_BINARY)
#cv2.namedWindow("threshold", cv2.WINDOW_NORMAL)
#cv2.imshow("threshold", thresh1)

imgTest = cv2.subtract(threshold1, cannyEdge)

# binary image
contours, _ = cv2.findContours(threshold1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
num_count = 0
for i, cnt in enumerate(contours):

    x, y = cnt[0, 0]
    x1, y1, w1, h1 = cv2.boundingRect(cnt)
    if cv2.contourArea(contours[i]) > 500:
        cv2.circle(newImg, (x1 + round(w1 / 2), y1 + round(h1 / 2)), 1, (0, 255, 255), 5)
        cv2.drawContours(newImg, [cnt], -1, (0, 0, 255), thickness=cv2.FILLED)


cv2.namedWindow("contourImg", cv2.WINDOW_NORMAL)
cv2.imshow("contourImg", imgTest)


cv2.namedWindow("Contour", cv2.WINDOW_NORMAL)
cv2.imshow("Contour", imgTest)

cv2.namedWindow("CannyEdgeDetection", cv2.WINDOW_NORMAL)
cv2.imshow("CannyEdgeDetection", cannyEdge)
#cv2.createTrackbar('Threshold1', 'CannyEdgeDetection', 0, 1000, update_threshold1)
#cv2.createTrackbar('Threshold2', 'CannyEdgeDetection', 0, 1000, update_threshold2)
#updateScreen()
#cv2.createTrackbar("contrast", "contrastbrightness", 0, 1000, update_contrast)
#cv2.createTrackbar('brightness', "contrastbrightness", 0, 1000, update_brightness)
#updateScreen2()
cv2.waitKey(0)
cv2.destroyAllWindows()
