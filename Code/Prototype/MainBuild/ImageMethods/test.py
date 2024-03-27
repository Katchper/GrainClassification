import copy

import cv2
import numpy as np

canny_thresh1 = 177
canny_thresh2 = 140
dilate_canny = 1

binary_thresh2 = 57

contrast = 79
brightness = 335

kernel_2 = np.ones((2, 2), np.uint8)

def updateScreen():
    tempMod1 = updateBrightnessContrast()
    cannyEdge = cv2.Canny(tempMod1, canny_thresh1, canny_thresh2)
    thickerEdge = cv2.dilate(cannyEdge, kernel_2, iterations=dilate_canny)
    ret, threshold1 = cv2.threshold(grayCopy, binary_thresh2, 255, cv2.THRESH_BINARY)
    imgTest = cv2.subtract(threshold1, thickerEdge)
    cv2.imshow("contours", imgTest)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(imgTest, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=2)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv2.watershed(newImg, markers)
    newImg[markers == -1] = [255, 0, 0]

    img2 = newImg.copy()
    markers1 = markers.astype(np.uint8)
    ret, m2 = cv2.threshold(markers1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(m2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        x1, y1, w1, h1 = cv2.boundingRect(c)
        #    img2 = img.copy()
        #    cv2.waitKey(0)
        cv2.drawContours(img2, c, -1, (0, 0, 255), 3)
        cv2.circle(img2, (x1 + round(w1 / 2), y1 + round(h1 / 2)), 1, (0, 255, 255), 5)

    cv2.imshow("watershed", img2)

    #return contoursOutlines(imgTest)
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
    testImg = copy.deepcopy(newImg)
    contours, _ = cv2.findContours(finalImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for i, cnt in enumerate(contours):
        x1, y1, w1, h1 = cv2.boundingRect(cnt)
        if cv2.contourArea(contours[i]) > 250:
            count += 1
            cv2.drawContours(testImg, [cnt], -1, (0, 0, 255), thickness=cv2.FILLED)
            cv2.circle(testImg, (x1 + round(w1 / 2), y1 + round(h1 / 2)), 1, (0, 255, 255), 5)
    cv2.imshow("final", testImg)

# Read the image
original_image = cv2.imread("C:/Users/Katch/Desktop/grain/broken/broken over 1_5mm001.tif", cv2.IMREAD_GRAYSCALE)
cropped_image = original_image[1255:3400, 160:2350]
grayCopy = copy.deepcopy(cropped_image)

colour_image = cv2.imread("C:/Users/Katch/Desktop/grain/broken/broken over 1_5mm001.tif")
cropped_image2 = colour_image[1255:3400, 160:2350]
newImg = copy.deepcopy(cropped_image2)

cv2.namedWindow("watershed", cv2.WINDOW_NORMAL)
cv2.imshow("watershed", newImg)

updateScreen()
cv2.waitKey(0)
cv2.destroyAllWindows()