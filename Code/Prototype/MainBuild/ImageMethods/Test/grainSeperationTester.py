"""
grainSeperationTester
Author: Kacper Dziedzic ktd1
Version: 1.1

Code to test out the several values corresponding to image preprocessing
Gives the used trackbars to edit the values, and they get updated immediately
"""

import copy
import cv2
import numpy as np

# the values that can get changed
canny_thresh1 = 115
canny_thresh2 = 208
dilate_canny = 1

binary_thresh1 = 57
binary_thresh2 = 57

erosion1 = 1
erosion2 = 1

contrast1 = 65
brightness1 = 190

contrast2 = 101
brightness2 = 203

max_area = 10000

# update for every global variable and updates the screen right after
def update_erosion1(erosionTemp):
    global erosion1
    erosion1 = erosionTemp
    updateScreen()


def update_erosion2(erosionTemp):
    global erosion2
    erosion2 = erosionTemp
    updateScreen()

def update_area(areaTemp):
    global max_area
    max_area = areaTemp
    updateScreen()

def update_contrast(contrast11):
    global contrast1
    contrast1 = contrast11
    updateScreen()

def update_contrast2(contrast11):
    global contrast2
    contrast2 = contrast11
    updateScreen()
def update_dilation(dilation1):
    global dilate_canny
    dilate_canny = dilation1
    updateScreen()


def update_brightness(brightness11):
    global brightness1
    brightness1 = brightness11
    updateScreen()

def update_brightness2(brightness11):
    global brightness2
    brightness2 = brightness11
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

"""
method which applies the new values to the preprocessing on the original image and displays the result
"""
def updateScreen():
    tempMod1 = updateBrightnessContrast(brightness1, contrast1)

    tempMod2 = updateBrightnessContrast(brightness2, contrast2)
    cv2.imshow("step1", tempMod1)
    #
    cannyEdge = cv2.Canny(tempMod1, canny_thresh1, canny_thresh2)
    thickerEdge = cv2.dilate(cannyEdge, kernel_2, iterations=dilate_canny)
    cv2.imshow("step2", thickerEdge)
    #
    ret3, th4 = cv2.threshold(tempMod2, binary_thresh1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #cv2.imshow("step3", th4)
    ret, threshold1 = cv2.threshold(tempMod2, binary_thresh2, 255, cv2.THRESH_BINARY)
    cv2.imshow("step3", threshold1)
    overlay = copy.deepcopy(th4)
    imgTest = cv2.subtract(threshold1, thickerEdge)
    inner = cv2.erode(imgTest, kernel_2, iterations=erosion1)
    cv2.imshow("step4", inner)

    cv2.imshow("step5", imgTest)
    erosion = cv2.erode(overlay, kernel_2, iterations=erosion2)
    cv2.imshow("step6", erosion)
    finalTogether = cv2.add(imgTest, inner)
    #cv2.imshow("contours", imgTest)
    return contoursOutlines(imgTest)
    #final2 = cv2.add(finalTogether, cannyEdge)
    # image1 = cv2.cvtColor(tempMod1, cv2.COLOR_BGR2RGB)
    # image2 = cv2.cvtColor(thickerEdge, cv2.COLOR_BGR2RGB)
    # image3 = cv2.cvtColor(threshold1, cv2.COLOR_BGR2RGB)

"""
updates brightness and contrast
"""
def updateBrightnessContrast(bright, contr):
    max = 255
    brightnesstemp = int((bright - 0) * (255 - (-255)) / (510 - 0) + (-255))
    contrasttemp = int((contr - 0) * (127 - (-127)) / (254 - 0) + (-127))
    alpha12 = (max - brightnesstemp) / 255
    gamma12 = brightnesstemp
    caltest = cv2.addWeighted(grayCopy, alpha12, grayCopy, 0, gamma12)

    Alpha22 = float(131 * (contrasttemp + 127)) / (127 * (131 - contrasttemp))
    Gamma22 = 127 * (1 - Alpha22)
    caltest = cv2.addWeighted(caltest, Alpha22, caltest, 0, Gamma22)
    return caltest

"""
obtains the contours of the grains and displays them. 
"""
def contoursOutlines(finalImage):
    testImg = copy.deepcopy(colourCopy)
    eroded_final = cv2.erode(finalImage, kernel_2)
    contours, _ = cv2.findContours(eroded_final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    print(len(contours))
    for i, cnt in enumerate(contours):
        x1, y1, w1, h1 = cv2.boundingRect(cnt)
        area = round(cv2.contourArea(contours[i]))
        if  area > 500 and area < max_area:
            rect = cv2.minAreaRect(cnt)
            """if rect[1][0] > rect[1][1]:
                print(rect[1][1]/rect[1][0])
            else:
                print(rect[1][0] / rect[1][1])
            #print((rect[1][0], rect[1][1]))
            print("----")"""
            count += 1
            cv2.drawContours(testImg, [cnt], -1, (0, 0, 255), thickness=cv2.FILLED)
            cv2.circle(testImg, (x1 + round(w1 / 2), y1 + round(h1 / 2)), 1, (0, 255, 255), 5)
    # print(count)
    #image4 = cv2.cvtColor(testImg, cv2.COLOR_BGR2RGB)
    cv2.imshow("final", testImg)
    """if 520 <= count <= 550:
        print(canny_thresh1, canny_thresh2, binary_thresh1, binary_thresh2, dilate_canny, erosion1, erosion2, contrast,
              brightness)
        print(count)"""
        #return canny_thresh1, canny_thresh2, binary_thresh1, binary_thresh2, dilate_canny, erosion1, erosion2, contrast, brightness, count
    #return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0


original_image = cv2.imread("C:/Users/Katch/Desktop/grain/broken/broken over 1_5mm001.tif", cv2.IMREAD_GRAYSCALE)
cropped_image = original_image[1260:3400, 170:2350]
grayCopy = copy.deepcopy(cropped_image)
# colour image
colour_image = cv2.imread("C:/Users/Katch/Desktop/grain/broken/broken over 1_5mm001.tif")
cropped_image2 = colour_image[1260:3400, 170:2350]

colourCopy = copy.deepcopy(cropped_image2)

kernel_2 = np.ones((2, 2), np.uint8)
updateScreen()
###
cv2.namedWindow("final", cv2.WINDOW_NORMAL)
cv2.imshow("final", cropped_image2)

"""cv2.namedWindow("contours", cv2.WINDOW_NORMAL)
cv2.imshow("contours", cropped_image2)"""
updateScreen()
# all the sliders
windowName = "slidersMenu"
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
cv2.resizeWindow(windowName, 500,500)
cv2.createTrackbar('Contrast1', windowName, contrast1, 300, update_contrast)
cv2.createTrackbar('Brightness1', windowName,  brightness1, 300, update_brightness)
cv2.createTrackbar('Contrast2', windowName, contrast2, 300, update_contrast2)
cv2.createTrackbar('Brightness2', windowName,  brightness2, 300, update_brightness2)
cv2.createTrackbar('Canny1', windowName, canny_thresh1, 500, update_threshold3)
cv2.createTrackbar('Canny2', windowName,  canny_thresh2, 500, update_threshold4)
cv2.createTrackbar('Dilation', windowName, dilate_canny, 5, update_dilation)
cv2.createTrackbar('Thresh2', windowName,  binary_thresh2, 255, update_threshold2)
cv2.createTrackbar('MaxArea', windowName,  max_area, 25000, update_area)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
pool method to get the total number of contours from update screena and output them to console.
"""
def poolMethod(contrastCount1, brightnessCount1, cannyCount11, cannyCount21, dilationCount1, erosCount11, erosCount21,
               threshCount11):
    global contrast1
    contrast = contrastCount1
    global brightness1
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


# testing method for getting optimal image preprocessing automatically
# this is however too inneficient due to the sheer scale of values.

"""if __name__ == '__main__':
    print("")"""

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