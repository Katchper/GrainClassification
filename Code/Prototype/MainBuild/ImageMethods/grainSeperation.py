import copy
import time
from multiprocessing import Pool

import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import sort

from Code.Prototype.MainBuild.FileMethods.arffBuilder import writeLineToARFF

canny_thresh1 = 50
canny_thresh2 = 200
dilate_canny = 1

binary_thresh1 = 20
binary_thresh2 = 255

erosion1 = 1
erosion2 = 1

contrast = 75
brightness = 225


# canny edge detection thresholds
# i need sliders for the contrast, brightness
# the 2 threshold values

# each change updates everything else
# shows contours and a dot in each contour

def update_erosion1(erosionTemp):
    global erosion1
    erosion1 = erosionTemp
    updateScreen()


def update_erosion2(erosionTemp):
    global erosion2
    erosion2 = erosionTemp
    updateScreen()


def update_contrast(contrast1):
    global contrast
    contrast = contrast1
    updateScreen()


def update_dilation(dilation1):
    global dilate_canny
    dilate_canny = dilation1
    updateScreen()


def update_brightness(brightness1):
    global brightness
    brightness = brightness1
    updateScreen()


def update_threshold1(threshold1):
    global binary_thresh1
    binary_thresh1 = threshold1
    updateScreen()


def update_threshold2(threshold2):
    global binary_thresh2
    binary_thresh2 = threshold2
    updateScreen()


def update_threshold3(threshold3):
    global canny_thresh1
    canny_thresh1 = threshold3
    updateScreen()


def update_threshold4(threshold4):
    global canny_thresh2
    canny_thresh2 = threshold4
    updateScreen()


def updateScreen():
    tempMod1 = updateBrightnessContrast()
    #
    cannyEdge = cv2.Canny(tempMod1, canny_thresh1, canny_thresh2)
    thickerEdge = cv2.dilate(cannyEdge, kernel_2, iterations=dilate_canny)
    #
    ret3, th4 = cv2.threshold(grayCopy, 40, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ret, threshold1 = cv2.threshold(grayCopy, 45, 255, cv2.THRESH_BINARY)
    overlay = copy.deepcopy(th4)
    inner = cv2.erode(overlay, kernel_2, iterations=erosion1)
    imgTest = cv2.subtract(threshold1, thickerEdge)
    erosion = cv2.erode(imgTest, kernel_2, iterations=erosion2)
    finalTogether = cv2.add(erosion, inner)
    return contoursOutlines(finalTogether)
    # final2 = cv2.add(finalTogether, cannyEdge)
    # image1 = cv2.cvtColor(tempMod1, cv2.COLOR_BGR2RGB)
    # image2 = cv2.cvtColor(thickerEdge, cv2.COLOR_BGR2RGB)
    # image3 = cv2.cvtColor(threshold1, cv2.COLOR_BGR2RGB)


def updateBrightnessContrast():
    max = 255
    brightnesstemp = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
    contrasttemp = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
    alpha12 = (max - brightnesstemp) / 255
    gamma12 = brightnesstemp
    caltest = cv2.addWeighted(grayCopy, alpha12, grayCopy, 0, gamma12)

    Alpha22 = float(131 * (contrasttemp + 127)) / (127 * (131 - contrasttemp))
    Gamma22 = 127 * (1 - Alpha22)
    caltest = cv2.addWeighted(caltest, Alpha22, caltest, 0, Gamma22)
    return caltest


def contoursOutlines(finalImage):
    testImg = copy.deepcopy(colourCopy)
    contours, _ = cv2.findContours(finalImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for i, cnt in enumerate(contours):
        x1, y1, w1, h1 = cv2.boundingRect(cnt)
        if cv2.contourArea(contours[i]) > 250:
            count += 1
            cv2.drawContours(testImg, [cnt], -1, (0, 0, 255), thickness=cv2.FILLED)
            cv2.circle(testImg, (x1 + round(w1 / 2), y1 + round(h1 / 2)), 1, (0, 255, 255), 5)
    # print(count)
    image4 = cv2.cvtColor(testImg, cv2.COLOR_BGR2RGB)
    plt.imshow(image4)
    plt.subplots_adjust(left=0, right=0.99, top=0.99, bottom=0, wspace=0, hspace=0.01)
    plt.show()
    if 520 <= count <= 550:
        """print(canny_thresh1, canny_thresh2, binary_thresh1, binary_thresh2, dilate_canny, erosion1, erosion2, contrast,
              brightness)
        print(count)"""
        #return canny_thresh1, canny_thresh2, binary_thresh1, binary_thresh2, dilate_canny, erosion1, erosion2, contrast, brightness, count
    #return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0


original_image = cv2.imread("C:/Users/Katch/Desktop/grain/broken/broken over 1_5mm001.tif", cv2.IMREAD_GRAYSCALE)
cropped_image = original_image[1255:3400, 160:2350]
grayCopy = copy.deepcopy(cropped_image)
# colour image
colour_image = cv2.imread("C:/Users/Katch/Desktop/grain/broken/broken over 1_5mm001.tif")
cropped_image2 = colour_image[1255:3400, 160:2350]

colourCopy = copy.deepcopy(cropped_image2)

kernel_2 = np.ones((2, 2), np.uint8)
# updateScreen()
###
plt.imshow(cropped_image2)
plt.subplots_adjust(left=0, right=0.99, top=0.99, bottom=0, wspace=0, hspace= 0.01)
# Show plot
plt.show()

windowName = "slidersMenu"
cv2.namedWindow(windowName)
cv2.createTrackbar('Contrast', windowName, 75, 1000, update_contrast)
cv2.createTrackbar('Brightness', windowName,  225, 1000, update_brightness)
cv2.createTrackbar('Canny1', windowName, 50, 1000, update_threshold3)
cv2.createTrackbar('Canny2', windowName,  200, 1000, update_threshold4)
cv2.createTrackbar('Dilation', windowName, 1, 5, update_dilation)
cv2.createTrackbar('Thresh1', windowName, 20, 500, update_threshold1)
cv2.createTrackbar('Thresh2', windowName,  255, 255, update_threshold2)
cv2.createTrackbar('FirstErosion', windowName, 1, 5, update_erosion1)
cv2.createTrackbar('ErosionOverlay', windowName,  1, 5, update_erosion2)
cv2.waitKey(1)
cv2.destroyAllWindows()

def poolMethod(contrastCount1, brightnessCount1, cannyCount11, cannyCount21, dilationCount1, erosCount11, erosCount21,
               threshCount11):
    global contrast
    contrast = contrastCount1
    global brightness
    brightness = brightnessCount1
    global canny_thresh1
    canny_thresh1 = cannyCount11
    global canny_thresh2
    canny_thresh2 = cannyCount21
    global dilate_canny
    dilate_canny = dilationCount1
    global binary_thresh1
    binary_thresh1 = threshCount11
    global erosion1
    erosion1 = erosCount11
    global erosion2
    erosion2 = erosCount21
    values = updateScreen()
    return values


def process(args):
    return poolMethod(*args)


# Plot images

if __name__ == '__main__':
    print("")

"""    arguments = []
    for contrastCount in range(50, 200, 25):
        for brightnessCount in range(100, 300, 25):
            for cannyCount1 in range(50, 250, 25):
                for cannyCount2 in range(50, 250, 25):
                    for dilationCount in range(2):
                        for erosCount1 in range(2):
                            for erosCount2 in range(2):
                                for threshCount1 in range(20, 60, 5):
                                    arguments.append((contrastCount, brightnessCount, cannyCount1, cannyCount2,
                                                      dilationCount, erosCount1, erosCount2, threshCount1))
    print("list done")
    with Pool() as pool:
        results = pool.map(process, arguments)
    #print(results)
    sorted_results = sorted(filter(lambda x: x is not None, results), key=lambda x: x[9], reverse=True)
    top_10 = sorted_results[:10]
    for item in top_10:
        print("------")
        print(item)"""

# brightness/contrast sliders
# slider for canny edge thresholds
# erosion iterations
# final threshold slider
# display final contour regions

# 0-256 contrast
# brightness 0-400
# 0-1000 canny1
# 0-1000 canny2
# dilation 0-5
# erosion1 0-10
# erosion2 0-10
# thresh1 - 0-500
# thresh2 - 0-500