"""
CannyEdgeTester
Author: Kacper Dziedzic ktd1
Version: 1.1

Code to test out preset values for image preprocessing, namely the canny edge.
"""

import copy
import cv2
import numpy as np

thresh1 = 0
thresh2 = 0
thresh3 = 0
thresh4 = 0
contrast = 0
brightness = 0

# canny edge detection thresholds
# i need sliders for the contrast, brightness
# the 2 threshold values

# each change updates everything else
# shows contours and a dot in each contour

"""
methods to update the global variables and update the screen
"""
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

def update_threshold3(threshold3):
    global thresh3
    thresh3 = threshold3
    updateScreen()
def update_threshold4(threshold4):
    global thresh4
    thresh4 = threshold4
    updateScreen()

"""
Updates the images displayed by using the values from the sliders.
"""
def updateScreen():
    cannyEdge = cv2.Canny(original_image[1255:3400, 160:2350], thresh3, thresh4)
   #cv2.imshow('CannyEdgeDetection', cannyEdge)
    imgTest = cv2.subtract(threshold1, cannyEdge)

    #ret, imgTest = cv2.threshold(out, thresh1, thresh2, cv2.THRESH_BINARY)
    cv2.imshow("Contour", imgTest)

"""
Contrast brightness updater
"""
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

original_image = cv2.imread("C:/Users/Katch/Desktop/grain/broken/broken over 1_5mm001.tif", cv2.IMREAD_GRAYSCALE)
cropped_image = original_image[1255:3400, 160:2350]

# colour image
colour_image = cv2.imread("C:/Users/Katch/Desktop/grain/broken/broken over 1_5mm001.tif")
cropped_image2 = colour_image[1255:3400, 160:2350]
newImg = copy.deepcopy(cropped_image2)

cv2.namedWindow("CannyEdgeDetection", cv2.WINDOW_NORMAL)
cv2.imshow("CannyEdgeDetection", cropped_image2)

out = copy.deepcopy(cropped_image)

kernel_1 = np.ones((1,1), np.uint8)                                       #dilation kernel
kernel_2 = np.ones((2,2), np.uint8)

cannyEdge = cv2.Canny(cropped_image2, 280, 155)
thickerEdge = cv2.dilate(cannyEdge, kernel_2, iterations=2)
"""
plt.subplot(121), plt.imshow(cropped_image2, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()"""

new_image = copy.deepcopy(cropped_image2)
# contrast + brightness changes

ret3, th4 = cv2.threshold(out, 40, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

ret, threshold1 = cv2.threshold(out, 45, 255, cv2.THRESH_BINARY)
cv2.namedWindow("threshold", cv2.WINDOW_NORMAL)
cv2.imshow("threshold", threshold1)
overlay = copy.deepcopy(th4)
inner = cv2.erode(overlay, kernel_2, iterations=2)
cv2.namedWindow("innards", cv2.WINDOW_NORMAL)
cv2.imshow("innards", thickerEdge)
imgTest = cv2.subtract(threshold1, thickerEdge)                                  #erosion kernel
                 #apply dialation
erosion = cv2.erode(imgTest, kernel_2, iterations=1)

#apply erosion
# binary image
finalTogether = cv2.add(erosion, inner)
final2 = cv2.add(finalTogether, cannyEdge)
contours, _ = cv2.findContours(finalTogether, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
num_count = 0
for i, cnt in enumerate(contours):

    x, y = cnt[0, 0]
    x1, y1, w1, h1 = cv2.boundingRect(cnt)
    if cv2.contourArea(contours[i]) > 250:
        cv2.drawContours(newImg, [cnt], -1, (0, 0, 255), thickness=cv2.FILLED)
        cv2.circle(newImg, (x1 + round(w1 / 2), y1 + round(h1 / 2)), 1, (0, 255, 255), 5)



cv2.namedWindow("contourImg", cv2.WINDOW_NORMAL)
cv2.imshow("contourImg", newImg)


cv2.namedWindow("Contour", cv2.WINDOW_NORMAL)
cv2.imshow("Contour", threshold1)

#cv2.createTrackbar('Threshold1', 'Contour', 0, 1000, update_threshold1)
#cv2.createTrackbar('Threshold2', 'Contour', 0, 1000, update_threshold2)

cv2.createTrackbar('Canny1', 'Contour', 0, 1000, update_threshold3)
cv2.createTrackbar('Canny2', 'Contour', 0, 1000, update_threshold4)

#updateScreen()
#cv2.createTrackbar("contrast", "contrastbrightness", 0, 1000, update_contrast)
#cv2.createTrackbar('brightness', "contrastbrightness", 0, 1000, update_brightness)
#updateScreen2()
cv2.waitKey(0)
cv2.destroyAllWindows()