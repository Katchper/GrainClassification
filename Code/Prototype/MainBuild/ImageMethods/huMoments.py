import time

import cv2
import numpy as np

from Code.Prototype.MainBuild.FileMethods.arffBuilder import writeLineToARFF


def process_image(file_dir, image_name, image_dir, grain_name, visual):
    original_image = cv2.imread(image_dir + image_name, cv2.IMREAD_GRAYSCALE)
    cropped_image = original_image[1245:3400, 160:2350]

    # colour image
    colour_image = cv2.imread(image_dir + image_name)
    cropped_image2 = colour_image[1245:3400, 160:2350]
    HSV_image = cv2.cvtColor(cropped_image2, cv2.COLOR_BGR2HSV)

    out = cv2.addWeighted(cropped_image, 1, cropped_image, 0, 0)

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

    # apply a threshold
    # ret3, th4 = cv2.threshold(cal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ret, thresh1 = cv2.threshold(cal, 65, 255, cv2.THRESH_BINARY)
    # binary image
    contours, _ = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    num_count = 0
    for i, cnt in enumerate(contours):
        x, y = cnt[0, 0]
        if cv2.contourArea(contours[i]) > 500:
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
            if visual:
                print("area: " + str(area) + " outline length: " + str(outline))
                cv2.drawContours(cropped_image2, [cnt], -1, (0, 0, 255), 3)
                cv2.putText(cropped_image2, f'Contour {num_count}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 0), 2)
                print(f"\nHuMoments for Contour {count}:\n", hm)
            writeLineToARFF(file_dir, total_hue, total_sat, total_val, area, outline, hm, grain_name)

    if visual:
        cv2.namedWindow('1', cv2.WINDOW_NORMAL)
        cv2.namedWindow('test', cv2.WINDOW_NORMAL)
        cv2.imshow('1', thresh1)
        cv2.imshow('test', cropped_image2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print("Image Complete")


visual_testing = False


# process_image(visual_testing)

def process_image_to_values(image_name, image_dir, grain_name):
    original_image = cv2.imread(image_dir + image_name, cv2.IMREAD_GRAYSCALE)
    cropped_image = original_image[1245:3400, 160:2350]

    # colour image
    colour_image = cv2.imread(image_dir + image_name)
    cropped_image2 = colour_image[1245:3400, 160:2350]
    HSV_image = cv2.cvtColor(cropped_image2, cv2.COLOR_BGR2HSV)

    out = cv2.addWeighted(cropped_image, 1, cropped_image, 0, 0)

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

    # apply a threshold
    # ret3, th4 = cv2.threshold(cal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ret, thresh1 = cv2.threshold(cal, 65, 255, cv2.THRESH_BINARY)
    # binary image
    contours, _ = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hue_list = []
    sat_list = []
    val_list = []
    area_list = []
    outline_list = []
    hm_list = []
    grain_name_list = []

    num_count = 0
    for i, cnt in enumerate(contours):
        if cv2.contourArea(contours[i]) > 500:
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
            hue_list.append(total_hue)
            sat_list.append(total_sat)
            val_list.append(total_val)
            area_list.append(area)
            outline_list.append(outline)
            hm_list.append(hm)
            grain_name_list.append(grain_name)

    return hue_list, sat_list, val_list, area_list, outline_list, hm_list, grain_name_list

def process_image_demo(image_name, image_dir):
    original_image = cv2.imread(image_dir + image_name, cv2.IMREAD_GRAYSCALE)
    cropped_image = original_image[1245:3400, 160:2350]

    # colour image
    colour_image = cv2.imread(image_dir + image_name)
    cropped_image2 = colour_image[1245:3400, 160:2350]

    out = cv2.addWeighted(cropped_image, 1, cropped_image, 0, 0)

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
    cv2.imshow("contrastbrightness", out)

    # apply a threshold
    # ret3, th4 = cv2.threshold(cal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ret, thresh1 = cv2.threshold(cal, 65, 255, cv2.THRESH_BINARY)
    cv2.namedWindow("threshold", cv2.WINDOW_NORMAL)
    cv2.imshow("threshold", thresh1)

    # binary image
    contours, _ = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    num_count = 0
    for i, cnt in enumerate(contours):
        x, y = cnt[0, 0]
        x1, y1, w1, h1 = cv2.boundingRect(cnt)
        if cv2.contourArea(contours[i]) > 500:
            cv2.circle(cropped_image2, (x1 + round(w1 / 2), y1 + round(h1 / 2)), 1, (0, 255, 255), 5)
            cv2.drawContours(cropped_image2, [cnt], -1, (0, 0, 255), 2)

    cv2.namedWindow("contours", cv2.WINDOW_NORMAL)
    cv2.imshow("contours", cropped_image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()