import cv2
import os
import numpy as np

# Read the original image


# get file directory
# get all image names within the directory
# create a method which returns a list of images from one main image
# create a method which loops through all the images within the file to obtain a final list
# do machine learning things
# use one of the test images to see whether it works

file_dir = "C:/Users/Katch/Desktop/Major Project/grain photos/groat1/"
images = os.listdir(file_dir)


def read_image(file_path, image_path):
    # binary image
    original_image = cv2.imread(file_path + image_path, cv2.IMREAD_GRAYSCALE)
    cropped_image = original_image[1230:3400, 160:2350]
    dst = cv2.resize(cropped_image, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
    # colour image
    colour_image = cv2.imread(file_path + image_path)
    cropped_image2 = colour_image[1230:3400, 160:2350]
    dst2 = cv2.resize(cropped_image2, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)

    # increase contrast
    out = cv2.addWeighted(dst, 1, dst, 0, 0)
    # apply a gaussian
    blur = cv2.GaussianBlur(out, (5, 5), 0)
    # do a otsu binary threshold
    ret3, th4 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # get contours
    contours, _ = cv2.findContours(th4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grain_list = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter out small noise (adjust the threshold as needed)
        if w * h > 1000:
            extracted_segment = cropped_image2[y:y + h, x:x + w].copy()
            grain_list.append(extracted_segment)
            cv2.rectangle(cropped_image2, (x,y), (x+w,y+h), (0,0,255), 4)

    #cv2.namedWindow('Grain Detection', cv2.WINDOW_NORMAL)
    #cv2.imshow('Grain Detection', cropped_image2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return grain_list


def getDataSet(images, data):
    data_set = []
    for img in images:
        print('Completed image')
        data_set.extend(read_image(file_dir, img))
    return data_set


def testDataSet():
    training_data = getDataSet(images, 'groats')

    cv2.imshow('grain1', training_data[2])
    cv2.imshow('grain2', training_data[2022])
    cv2.imshow('grain3', training_data[400])

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(training_data)


testDataSet()
# for each image in a folder
# get an image, get all the coordinates for every individual grain for the image
# for each grain in the array of grains

# get a contour of the grain, remove the background

# obtain the HSV colour values for each pixel within the grain.
# counter increase for each pixel - final count = area
# calculate the average HSV colour for the grain

# place the area and colour into 2 separate arrays.

# what features/information can help distinguish between grain types:

# area/size, colour,
# circularity & shape,
# common features like the hairs on wholegrain.

# make a bell curve graph for the areas, colours, circularity value.
# Show difference between grain types
