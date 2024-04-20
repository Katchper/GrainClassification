import copy
import math
import time

import cv2
import numpy as np

from Code.Prototype.MainBuild.FileMethods.arffBuilder import writeLineToARFF

def updateBrightnessContrast(grayCopy, bright, contr):
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

def process_image(file_dir, image_name, image_dir, grain_name, visual):
    if grain_name == "wholegrain":
        max_area = 8000
        min_area = 500
        canny_thresh1 = 115
        canny_thresh2 = 208
        dilate_canny = 1

        binary_thresh1 = 57

        contrast1 = 65
        brightness1 = 190

        contrast2 = 101
        brightness2 = 203
    else:
        max_area = 4600
        min_area = 250
        canny_thresh1 = 182
        canny_thresh2 = 90
        dilate_canny = 1

        binary_thresh1 = 48

        contrast1 = 51
        brightness1 = 162

        contrast2 = 101
        brightness2 = 203

    kernel_2 = np.ones((2, 2), np.uint8)

    original_image = cv2.imread(image_dir + image_name, cv2.IMREAD_GRAYSCALE)
    if image_dir == "C:/Users/Katch/Desktop/grain/broken2/":
        cropped_image = original_image[1680:4600, 210:3110]
        greyImage = copy.deepcopy(cropped_image)
        # colour image
        colour_image = cv2.imread(image_dir + image_name)
        cropped_image2 = colour_image[1680:4600, 210:3110]
    else:
        cropped_image = original_image[1240:3450, 160:2350]
        greyImage = copy.deepcopy(cropped_image)
        # colour image
        colour_image = cv2.imread(image_dir + image_name)
        cropped_image2 = colour_image[1240:3450, 160:2350]
    HSV_image = cv2.cvtColor(cropped_image2, cv2.COLOR_BGR2HSV)

    # contrast + brightness changes
    modifImage1 = updateBrightnessContrast(greyImage, brightness1, contrast1)
    modifImage2 = updateBrightnessContrast(greyImage, brightness2, contrast2)

    cannyEdge = cv2.Canny(modifImage1, canny_thresh1, canny_thresh2)
    thickerEdge = cv2.dilate(cannyEdge, kernel_2, iterations=dilate_canny)
    ret, threshold1 = cv2.threshold(modifImage2, binary_thresh1, 255, cv2.THRESH_BINARY)
    imgTest = cv2.subtract(threshold1, thickerEdge)

    # apply a threshold
    # ret3, th4 = cv2.threshold(cal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # binary image
    contours, _ = cv2.findContours(imgTest, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    num_count = 0
    for i, cnt in enumerate(contours):
        x, y = cnt[0, 0]
        area_temp = round(cv2.contourArea(contours[i]))
        if min_area < area_temp < max_area:
            mask = np.zeros_like(cropped_image2, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], 0, (255, 255, 255), thickness=cv2.FILLED)
            result = cv2.bitwise_and(HSV_image, mask)
            x1, y1, w, h = cv2.boundingRect(cnt)
            final_section = result[y1:y1 + h, x1:x1 + w]
            total_hue = 0
            total_sat = 0
            total_val = 0
            count = 0
            for n in range(final_section.shape[0]):
                for m in range(final_section.shape[1]):
                    hue, saturation, value = final_section[n, m]
                    if value != 0:
                        count += 1
                        total_hue += hue
                        total_sat += saturation
                        total_val += value

            # print(result)
            # cv2.imshow("f", result)
            total_pixels = count
            total_hue = total_hue / total_pixels
            total_sat = total_sat / total_pixels
            total_val = total_val / total_pixels
            if visual:
                print("pixels done = " + str(total_pixels))
                print(str(total_hue) + " " + str(total_sat) + " " + str(total_val))
            num_count += 1
            moments = cv2.moments(cnt)
            hm = cv2.HuMoments(moments)
            # apply log transformation so that numbers are on the same scale
            for j in range(0, 7):
                hm[j] = -1 * np.copysign(1.0, hm[j]) * np.log10(abs(hm[j]))

            # send hm, area, outline, total_hue, total_sat, total_val to write data.
            # 6 hu moment values - used for shape similarity
            # area value
            # outline length
            # average colour in HSV
            area = cv2.contourArea(cnt)
            outline = cv2.arcLength(cnt, True)
            outline = round(outline, 2)

            circularity = 4 * math.pi * (area / (outline * outline))
            rect = cv2.minAreaRect(cnt)
            rectangularity = area / (rect[1][0] * rect[1][1])
            if visual:
                print("area: " + str(area) + " outline length: " + str(outline))
                cv2.drawContours(cropped_image2, [cnt], -1, (0, 0, 255), 3)
                cv2.putText(cropped_image2, f'Contour {num_count}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 0), 2)
                print(f"\nHuMoments for Contour {count}:\n", hm)
            writeLineToARFF(file_dir, total_hue, total_sat, total_val, area, outline, hm, circularity, rectangularity, grain_name)

    if visual:
        cv2.namedWindow('1', cv2.WINDOW_NORMAL)
        cv2.namedWindow('test', cv2.WINDOW_NORMAL)
        cv2.imshow('1', imgTest)
        cv2.imshow('test', cropped_image2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print("Image Complete")


visual_testing = False


# process_image_to_values is the method main uses

def process_image_to_values(image_name, image_dir, grain_name):
    #ticcy = time.perf_counter()
    if grain_name == "wholegrain":
        max_area = 8000
        min_area = 500
        canny_thresh1 = 115
        canny_thresh2 = 208
        dilate_canny = 1

        binary_thresh1 = 57

        contrast1 = 65
        brightness1 = 190

        contrast2 = 101
        brightness2 = 203
    else:
        max_area = 4600
        min_area = 250
        canny_thresh1 = 182
        canny_thresh2 = 90
        dilate_canny = 1

        binary_thresh1 = 48

        contrast1 = 51
        brightness1 = 162

        contrast2 = 101
        brightness2 = 203
    kernel_2 = np.ones((2, 2), np.uint8)

    original_image = cv2.imread(image_dir + image_name, cv2.IMREAD_GRAYSCALE)

    if image_dir == "C:/Users/Katch/Desktop/grain/broken2/":
        cropped_image = original_image[1680:4600, 210:3110]
        # colour image
        colour_image = cv2.imread(image_dir + image_name)
        cropped_image2 = colour_image[1680:4600, 210:3110]
    else:
        cropped_image = original_image[1240:3450, 160:2350]
        # colour image
        colour_image = cv2.imread(image_dir + image_name)
        cropped_image2 = colour_image[1240:3450, 160:2350]

    greyImage = copy.deepcopy(cropped_image)
    HSV_image = cv2.cvtColor(cropped_image2, cv2.COLOR_BGR2HSV)

    # contrast + brightness changes
    modifImage1 = updateBrightnessContrast(greyImage, brightness1, contrast1)
    modifImage2 = updateBrightnessContrast(greyImage, brightness2, contrast2)

    cannyEdge = cv2.Canny(modifImage1, canny_thresh1, canny_thresh2)
    thickerEdge = cv2.dilate(cannyEdge, kernel_2, iterations=dilate_canny)
    ret, threshold1 = cv2.threshold(modifImage2, binary_thresh1, 255, cv2.THRESH_BINARY)
    imgTest = cv2.subtract(threshold1, thickerEdge)
    # binary image
    eroded_final = cv2.erode(imgTest, kernel_2)
    contours, _ = cv2.findContours(eroded_final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)


    hue_list = []
    sat_list = []
    val_list = []
    hm_list = []
    grain_name_list = []
    circularity_list = []
    circularity2_list = []
    rectangularity_list = []
    aspect_ratio_list = []
    compact_list = []

    num_count = 0
    for i, cnt in enumerate(contours):
        #print(len(contours))
        area_temp = round(cv2.contourArea(contours[i]))
        if min_area < area_temp < max_area:
            mask = np.zeros_like(cropped_image2, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], 0, (255, 255, 255), thickness=cv2.FILLED)
            result = cv2.bitwise_and(HSV_image, mask)
            x1, y1, w, h = cv2.boundingRect(cnt)
            final_section = result[y1:y1 + h, x1:x1 + w]
            total_hue = 0
            total_sat = 0
            total_val = 0
            count = 0
            for n in range(final_section.shape[0]):
                for m in range(final_section.shape[1]):
                    hue, saturation, value = final_section[n, m]
                    if value != 0:
                        count += 1
                        total_hue += hue
                        total_sat += saturation
                        total_val += value

            # print(result)
            # cv2.imshow("f", result)
            total_pixels = count
            total_hue = total_hue / total_pixels
            total_sat = total_sat / total_pixels
            total_val = total_val / total_pixels

            num_count += 1
            moments = cv2.moments(cnt)
            hm = cv2.HuMoments(moments)
            # apply log transformation so that numbers are on the same scale
            for j in range(0, 7):
                hm[j] = -1 * np.copysign(1.0, hm[j]) * np.log10(abs(hm[j]))

            # send hm, area, outline, total_hue, total_sat, total_val to write data.
            # 6 hu moment values - used for shape similarity
            # area value
            # outline length
            # average colour in HSV
            area = cv2.contourArea(cnt)
            outline = cv2.arcLength(cnt, True)
            outline = round(outline, 2)
            circularity = 4 * math.pi * (area / (outline * outline))
            rect = cv2.minAreaRect(cnt)
            if rect[1][0] > rect[1][1]:
                aspect_ratio = rect[1][1]/rect[1][0]
            else:
                aspect_ratio = rect[1][0]/rect[1][1]

            rectangularity = area / (rect[1][0] * rect[1][1])

            compactness = area / outline

            (x,y), radius = cv2.minEnclosingCircle(cnt)
            min_circle_area = math.pi * (int(radius) * int(radius))

            circleRatio = min_circle_area / area

            compact_list.append(compactness)
            hue_list.append(total_hue)
            sat_list.append(total_sat)
            val_list.append(total_val)
            hm_list.append(hm)
            rectangularity_list.append(rectangularity)
            circularity_list.append(circularity)
            aspect_ratio_list.append(aspect_ratio)
            grain_name_list.append(grain_name)
            circularity2_list.append(circleRatio)
    #toccy = time.perf_counter()
    #print(f"grain done in {toccy - ticcy:0.4f} seconds")
    return hue_list, sat_list, val_list, hm_list, circularity_list, circularity2_list, rectangularity_list, aspect_ratio_list, compact_list, grain_name_list

def process_image_demo(image_name, image_dir, grain_name):
    print(image_dir)
    original_image = cv2.imread(image_dir + image_name, cv2.IMREAD_GRAYSCALE)
    if image_dir == "C:/Users/Katch/Desktop/grain/broken2/":
        print("true")
        cropped_image = original_image[1680:4600, 210:3110]
        # colour image
        colour_image = cv2.imread(image_dir + image_name)
        cropped_image2 = colour_image[1680:4600, 210:3110]
    else:
        cropped_image = original_image[1240:3450, 160:2350]
        print("false")
        # colour image
        colour_image = cv2.imread(image_dir + image_name)
        cropped_image2 = colour_image[1240:3450, 160:2350]

    greyImage = copy.deepcopy(cropped_image)

    cv2.namedWindow("contours", cv2.WINDOW_NORMAL)
    cv2.imshow("contours", cropped_image2)
    out = cv2.addWeighted(cropped_image, 1, cropped_image, 0, 0)

    if grain_name == "wholegrain":
        min_area = 500
        canny_thresh1 = 115
        canny_thresh2 = 208
        dilate_canny = 1

        binary_thresh1 = 57

        contrast1 = 65
        brightness1 = 190

        contrast2 = 101
        brightness2 = 203
    else:
        min_area = 250
        canny_thresh1 = 182
        canny_thresh2 = 90
        dilate_canny = 1

        binary_thresh1 = 48

        contrast1 = 51
        brightness1 = 162

        contrast2 = 101
        brightness2 = 203
    kernel_2 = np.ones((2, 2), np.uint8)

    cv2.namedWindow("contrastbrightness", cv2.WINDOW_NORMAL)
    cv2.imshow("contrastbrightness", out)

    # apply a threshold
    # ret3, th4 = cv2.threshold(cal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    HSV_image = cv2.cvtColor(cropped_image2, cv2.COLOR_BGR2HSV)

    # contrast + brightness changes
    modifImage1 = updateBrightnessContrast(greyImage, brightness1, contrast1)
    modifImage2 = updateBrightnessContrast(greyImage, brightness2, contrast2)

    cannyEdge = cv2.Canny(modifImage1, canny_thresh1, canny_thresh2)
    thickerEdge = cv2.dilate(cannyEdge, kernel_2, iterations=dilate_canny)
    ret, threshold1 = cv2.threshold(modifImage2, binary_thresh1, 255, cv2.THRESH_BINARY)
    imgTest = cv2.subtract(threshold1, thickerEdge)

    cv2.namedWindow("threshold", cv2.WINDOW_NORMAL)
    cv2.imshow("threshold", imgTest)

    # binary image
    contours, _ = cv2.findContours(imgTest, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    num_count = 0
    for i, cnt in enumerate(contours):
        x, y = cnt[0, 0]
        x1, y1, w1, h1 = cv2.boundingRect(cnt)
        if cv2.contourArea(contours[i]) > 500:
            cv2.circle(cropped_image2, (x1 + round(w1 / 2), y1 + round(h1 / 2)), 1, (0, 255, 255), 5)
            cv2.drawContours(cropped_image2, [cnt], -1, (0, 0, 255), 2)

    cv2.imshow("contours", cropped_image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()