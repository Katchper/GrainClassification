import cv2
import os
import numpy as np


# open the file with w to overwrite, x to create only if there isn't an existing one.

# HSV value ranges are 0-179, 0-255, 0-255
# to check on online colour picker multiply H by 2, divide the other 2 by 2.55
def createARFF(file_dir):
    file = open(file_dir, "w")
    file.write("@RELATION ImageDataset\n")
    file.write("\n")
    file.write("@ATTRIBUTE hue NUMERIC\n")
    file.write("@ATTRIBUTE saturation NUMERIC\n")
    file.write("@ATTRIBUTE value NUMERIC\n")
    file.write("@ATTRIBUTE area NUMERIC\n")
    file.write("@ATTRIBUTE perimeter NUMERIC\n")
    file.write("@ATTRIBUTE hueMoment1 NUMERIC\n")
    file.write("@ATTRIBUTE hueMoment2 NUMERIC\n")
    file.write("@ATTRIBUTE hueMoment3 NUMERIC\n")
    file.write("@ATTRIBUTE hueMoment4 NUMERIC\n")
    file.write("@ATTRIBUTE hueMoment5 NUMERIC\n")
    file.write("@ATTRIBUTE hueMoment6 NUMERIC\n")
    file.write("@ATTRIBUTE hueMoment7 NUMERIC\n")
    file.write("@ATTRIBUTE class {wholegrain, groats, broken}\n")
    file.write("\n")
    file.write("@DATA\n")
    file.close()


def writeLineToARFF(file_dir, hue, sat, val, area, perimeter, hue_moments, grain):
    file = open(file_dir, "a")
    # here is where the data goes in the format: red, blue, green, area, grainType
    # example is: file.write("45, 32, 67, 1200, groat\n")
    text = (str(hue)
            + ", " + str(sat)
            + ", " + str(val)
            + ", " + str(area)
            + ", " + str(perimeter)
            + ", " + str(hue_moments[0][0])
            + ", " + str(hue_moments[1][0])
            + ", " + str(hue_moments[2][0])
            + ", " + str(hue_moments[3][0])
            + ", " + str(hue_moments[4][0])
            + ", " + str(hue_moments[5][0])
            + ", " + str(hue_moments[6][0])
            + ", " + grain
            + "\n")
    file.write(text)
    file.close()


def writeArffFromArray(file_dir, hue, sat, val, area, perimeter, hue_moments, grain):
    file = open(file_dir, "w")
    file.write("@RELATION ImageDataset\n")
    file.write("\n")
    file.write("@ATTRIBUTE hue NUMERIC\n")
    file.write("@ATTRIBUTE saturation NUMERIC\n")
    file.write("@ATTRIBUTE value NUMERIC\n")
    file.write("@ATTRIBUTE area NUMERIC\n")
    file.write("@ATTRIBUTE perimeter NUMERIC\n")
    file.write("@ATTRIBUTE hueMoment1 NUMERIC\n")
    file.write("@ATTRIBUTE hueMoment2 NUMERIC\n")
    file.write("@ATTRIBUTE hueMoment3 NUMERIC\n")
    file.write("@ATTRIBUTE hueMoment4 NUMERIC\n")
    file.write("@ATTRIBUTE hueMoment5 NUMERIC\n")
    file.write("@ATTRIBUTE hueMoment6 NUMERIC\n")
    file.write("@ATTRIBUTE hueMoment7 NUMERIC\n")
    file.write("@ATTRIBUTE class {wholegrain, groats, broken}\n")
    file.write("\n")
    file.write("@DATA\n")

    count = 0
    for current in hue:
        text = (str(current)
                + ", " + str(sat[count])
                + ", " + str(val[count])
                + ", " + str(area[count])
                + ", " + str(perimeter[count])
                + ", " + str(hue_moments[count][0][0])
                + ", " + str(hue_moments[count][1][0])
                + ", " + str(hue_moments[count][2][0])
                + ", " + str(hue_moments[count][3][0])
                + ", " + str(hue_moments[count][4][0])
                + ", " + str(hue_moments[count][5][0])
                + ", " + str(hue_moments[count][6][0])
                + ", " + grain[count]
                + "\n")
        file.write(text)
        count += 1

    file.close()
