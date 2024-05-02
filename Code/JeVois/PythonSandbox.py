import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois
import cv2
import numpy as np
import copy
import math

## Jevois module for Grain Detection
#
# This module is made to test the grain detection code tested within the main project files.
# Since weka isn't available on the JeVois camera an OpenCv machine learning model has been used
# Model = random trees
# Author: Kacper Dziedzic ktd1
# Version: 1.1
# The module is quite slow due to the resources the machine learning model uses. To get started ensure the JeVois camera
# starts on a different model before switching to this one.
##


# method used in the main component used to update the contrast and brightness of an image.
# returns the modified image.
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

class PythonSandbox:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("sandbox", 100, jevois.LOG_INFO)
        xml_path = pyjevois.share + "/grain/random_trees_model.xml"
        self.ML_model = cv2.ml.RTrees_load(xml_path)
        self.kernel_1 = np.ones((2, 2), np.uint8)
        self.canny_thresh1 = 115
        self.canny_thresh2 = 208
        self.dilate_canny = 1
        self.binary_thresh1 = 57
        self.contrast1 = 65
        self.brightness1 = 190
        self.contrast2 = 101
        self.brightness2 = 203
        self.min_area = 600
        
    # ###################################################################################################
    ## Process function with USB output
    def process(self, inframe, outframe):
        inimg = inframe.getCvRGBA()
        inimggray = inframe.getCvGRAY()
        BGR_image = cv2.cvtColor(inimg, cv2.COLOR_RGBA2BGR)
        HSV_image = cv2.cvtColor(inimg, cv2.COLOR_BGR2HSV)
        cropped_image2 = copy.deepcopy(inimg)
        
        self.timer.start()

        modifImage1 = updateBrightnessContrast(inimggray, self.brightness1, self.contrast1)
        modifImage2 = updateBrightnessContrast(inimggray, self.brightness2, self.contrast2)

        cannyEdge = cv2.Canny(modifImage1, self.canny_thresh1, self.canny_thresh2)
        thickerEdge = cv2.dilate(cannyEdge, self.kernel_1, iterations=self.dilate_canny)
        ret, threshold1 = cv2.threshold(modifImage2, self.binary_thresh1, 255, cv2.THRESH_BINARY)
        imgTest = cv2.subtract(threshold1, thickerEdge)
        eroded_final = cv2.erode(imgTest, self.kernel_1)
        
        contours, _ = cv2.findContours(eroded_final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        hull_list = []
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            hull_list.append(hull)
        
        num_count = 0
        for i, cnt in enumerate(contours):
        # print(len(contours))
            area_temp = round(cv2.contourArea(contours[i]))
            if self.min_area < area_temp:
                mask = np.zeros_like(HSV_image, dtype=np.uint8)
                cv2.drawContours(mask, [cnt], 0, (255, 255, 255), thickness=cv2.FILLED)
                result = cv2.bitwise_and(HSV_image, mask)
                x1, y1, w, h = cv2.boundingRect(cnt)
                final_section = result[y1:y1 + h, x1:x1 + w]

                non_zero_pixels = np.count_nonzero(final_section[:, :, 2])
                total_hue = np.sum(final_section[:, :, 0])
                total_sat = np.sum(final_section[:, :, 1])
                total_val = np.sum(final_section[:, :, 2])

                if non_zero_pixels != 0:
                    total_hue /= non_zero_pixels
                    total_sat /= non_zero_pixels
                    total_val /= non_zero_pixels
                 
                 
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
                    aspect_ratio = rect[1][1] / rect[1][0]
                else:
                    aspect_ratio = rect[1][0] / rect[1][1]

                rectangularity = area / (rect[1][0] * rect[1][1])

                compactness = area / outline

                (x, y), radius = cv2.minEnclosingCircle(cnt)
                min_circle_area = math.pi * (int(radius) * int(radius))

                circleRatio = min_circle_area / area
                
                cv2.circle(cropped_image2, (x1 + round(w / 2), y1 + round(h / 2)), 1, (255, 0, 0), 3)
                cv2.drawContours(cropped_image2, [cnt], -1, (0, 0, 255), 2)
                
                finalList = [[total_hue, total_sat, total_val, hm[0], hm[1], hm[2], hm[3], hm[4], hm[5], hm[6], circularity, circleRatio, rectangularity, aspect_ratio, compactness]]
                
                #cv2.putText(cropped_image2, str(finalList), (x1, y1),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2) 
                _, predictions = self.ML_model.predict(np.array(finalList, dtype=np.float32))
                if predictions[0][0] == 2.0:
                    cv2.putText(cropped_image2, "whole", (x1, y1+20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif predictions[0][0] == 1.0:
                    cv2.putText(cropped_image2, "groat", (x1, y1+20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(cropped_image2, "broken", (x1, y1+20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        
        # Write a title:
        # Write frames/s info from our timer into the edge map (NOTE: does not account for output conversion time):
        fps = self.timer.stop()
        cv2.putText(cropped_image2, str(fps), (0, 14),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Convert our OpenCv output image to video output format and send to host over USB:
        outframe.sendCvRGBA(cropped_image2)

    # ###################################################################################################
    ## Process function with GUI output
    def processGUI(self, inframe, helper):
        # Start a new display frame, gets its size and also whether mouse/keyboard are idle:
        idle, winw, winh = helper.startFrame()

        # Draw full-resolution input frame from camera:
        x, y, w, h = helper.drawInputFrame("c", inframe, False, False)
        
        # Get the next camera image (may block until it is captured):
        #inimg = inframe.getCvBGR()
        
        # Start measuring image processing time (NOTE: does not account for input conversion time):
        self.timer.start()

        # Some drawings:
        helper.drawCircle(50, 50, 20, 0xffffffff, True)
        helper.drawRect(100, 100, 300, 200, 0xffffffff, True)
        
        # Write frames/s info from our timer:
        fps = self.timer.stop()
        helper.iinfo(inframe, fps, winw, winh);

        # End of frame:
        helper.endFrame()
        
