import cv2
import os

from Code.Prototype.MainBuild.FileMethods.arffBuilder import createARFF
from Code.Prototype.MainBuild.imageProcessing import read_image
from Code.Prototype.MainBuild.trainingDataClass import TrainingData

# Read the original image


# get file directory
# get all image names within the directory
# create a method which returns a list of images from one main image
# create a method which loops through all the images within the file to obtain a final list
# do machine learning things
# use one of the test images to see whether it works
training_list = []
training_list.append(TrainingData("C:/Users/Katch/Desktop/grain photos/wholegrain/", "wholegrain"))
training_list.append(TrainingData("C:/Users/Katch/Desktop/grain photos/groat1/", "groats"))
training_list.append(TrainingData("C:/Users/Katch/Desktop/grain photos/groat2/", "groats"))
training_list.append(TrainingData("C:/Users/Katch/Desktop/grain photos/groat3/", "groats"))
training_list.append(TrainingData("C:/Users/Katch/Desktop/grain photos/broken/", "broken"))

# get a list of all the images within the files into the objects
for file in training_list:
    images = []
    for item in os.listdir(file.file_dir):
        if item.endswith(".tif"):
            images.append(item)
    file.image_list = images

# for each file, get a list of colours, area and then make a list of the grain type based on the list size
training_colours = []
training_sizes = []
training_grain_name = []

for file in training_list:
    print("starting file")
    for image in file.image_list:
        colours, area, names = read_image(file.file_dir, image, file.grain_type)
        print(len(colours))
        training_colours.extend(colours)
        training_sizes.extend(area)
        training_grain_name.extend(names)

# using the lists, create an arff file containing that data
createARFF("TrainingData/training_data.arff", training_colours, training_sizes, training_grain_name)

# a function used for testing
def testImage():
    images_test = read_image(training_list[1].file_dir, "21QC_002.tif")
    for image in images_test:
        height, width, _ = image.shape

    cv2.imshow('grain1', images_test[1])
    cv2.imshow('grain2', images_test[2])
    cv2.imshow('grain3', images_test[12])
    cv2.waitKey(0)
    cv2.destroyAllWindows()



#testImage()

# step by step for part 1 of demo
# get image from the testing file, extract the grains from the image.
# for each grain get the values and parse them through the machine learning model