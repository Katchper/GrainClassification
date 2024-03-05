import cv2
import os
import numpy as np


# open the file with w to overwrite, x to create only if there isn't an existing one.

# HSV value ranges are 0-179, 0-255, 0-255
# to check on online colour picker multiply H by 2, divide the other 2 by 2.55
def createARFF(colour_list, area_list, grain_list):
    file = open("TrainingData/training_data.arff", "w")
    file.write("@RELATION ImageDataset\n")
    file.write("\n")
    file.write("@ATTRIBUTE hue NUMERIC\n")
    file.write("@ATTRIBUTE saturation NUMERIC\n")
    file.write("@ATTRIBUTE value NUMERIC\n")
    file.write("@ATTRIBUTE area NUMERIC\n")
    file.write("@ATTRIBUTE class {wholegrain, groats, broken}\n")
    file.write("\n")
    file.write("@DATA\n")

    # here is where the data goes in the format: red, blue, green, area, grainType
    # example is: file.write("45, 32, 67, 1200, groat\n")
    count = 0
    for colour in colour_list:
        text = (str(colour[0])
                + ", " + str(colour[1])
                + ", " + str(colour[2])
                + ", " + str(area_list[count])
                + ", " + grain_list[count]
                + "\n")
        file.write(text)
        count += 1

    file.close()

