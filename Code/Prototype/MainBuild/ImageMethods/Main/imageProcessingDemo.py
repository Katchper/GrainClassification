"""
imageProcessingDemo
Author: Kacper Dziedzic ktd1
Version: 1.1

Methods which take in a test image to extract values from, send to ML model
to be classified and display the data visually onto the original image
"""

import cv2


# this method will read a file path and image name and return a list of all the grain images within
# the image as an array.
def read_image(file_path, image_path, grain_name):
    # binary image
    original_image = cv2.imread(file_path + image_path, cv2.IMREAD_GRAYSCALE)
    cropped_image = original_image[1230:3400, 160:2350]

    # colour image
    colour_image = cv2.imread(file_path + image_path)
    cropped_image2 = colour_image[1230:3400, 160:2350]
    HSV_image = cv2.cvtColor(cropped_image2, cv2.COLOR_BGR2HSV)

    # increase contrast
    out = cv2.addWeighted(cropped_image, 1, cropped_image, 0, 0)
    # apply a gaussian
    blur = cv2.GaussianBlur(out, (5, 5), 0)
    # do a otsu binary threshold
    ret3, th4 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # get contours
    contours, _ = cv2.findContours(th4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grain_list = []
    area_list = []
    colour_list = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter out small noise (adjust the threshold as needed)
        if w * h > 1000:
            # adds the area and colour to the lists
            area_list.append(w * h)
            colour_list.append(HSV_image[y+round(h/2), x+round(w/2)])
            grain_list.append(grain_name)
            #draw dot at each centre point
            #cv2.circle(cropped_image2, (x+round(w/2), y+round(h/2)), 1, (0,0,255), 4)
            # here i get the contour of the grain and workout the true area and center colour
            #extracted_segment = cropped_image2[y:y + h, x:x + w].copy()
            #grain_list.append(extracted_segment)
            #cv2.rectangle(cropped_image2, (x,y), (x+w,y+h), (0,0,255), 4)
    #print(colour_list)
    #cv2.namedWindow('Grain Detection', cv2.WINDOW_NORMAL)
    #cv2.imshow('Grain Detection', cropped_image2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return colour_list, area_list, grain_list

# go through every image in the folder to return a final array of all the images.
def getDataSet(images, data):
    data_set = []
    for img in images:
        print('Completed image')
        data_set.extend(read_image(data, img))
    return data_set

# get all the images and display 3 example grains.
def testDataSet():
    training_data = getDataSet('groats')

    cv2.imshow('grain1', training_data[2])
    cv2.imshow('grain2', training_data[2022])
    cv2.imshow('grain3', training_data[400])

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(training_data)

# get one grains colour values and return it.
# it currently gets the centre.
def getGrainValues(images_test):
    for image in images_test:
        height, width, _ = image.shape
        # put area of image into list
        area = height * width
        area_list = []
        area_list.append(area)
        # get the centre of the image
    return 0

def grainContour(image):
    return

def displayImage(file_path, image_path, predictions):
    original_image = cv2.imread(file_path + image_path, cv2.IMREAD_GRAYSCALE)
    cropped_image = original_image[1245:3400, 160:2350]

    # colour image
    colour_image = cv2.imread(file_path + image_path)
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
    count = 0
    wgrain = 0
    ggrain = 0
    bgrain = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 500:
            cv2.circle(cropped_image2, (x + round(w / 2), y + round(h / 2)), 1, (0, 0, 255), 4)
            #cv2.rectangle(cropped_image2, (x, y), (x + w, y + h), (0, 0, 255), 4)
            cv2.drawContours(cropped_image2, [contour], -1, (0, 0, 255), 3)
            text = str(predictions[count])
            if predictions[count] == "wholegrain":
                wgrain+=1
            elif predictions[count] == "groats":
                ggrain+=1
            else: bgrain += 1

            if text == "wholegrain":
                text = "w"
            elif text == "groats":
                text = "g"
            else:
                text = "b"
            cv2.putText(cropped_image2, text, (x + round(w / 2), y + round(h / 2)),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            count+=1

    print("Wholegrain count: " + str(wgrain))
    print("Groats count: " + str(ggrain))
    print("Broken count: " + str(bgrain))
    print("Total = " + str(count))
    cv2.namedWindow(image_path, cv2.WINDOW_NORMAL)
    cv2.imshow(image_path, cropped_image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

