from random import randint

import cv2
import os
import threading

from Code.Prototype.MainBuild.FileMethods.arffBuilder import createARFF, writeLineToARFF
from Code.Prototype.MainBuild.ImageMethods.huMoments import process_image
from Code.Prototype.MainBuild.imageProcessing import read_image, displayImage
from Code.Prototype.MainBuild.machineLearning import machineLearningAlgorithm
from Code.Prototype.MainBuild.trainingDataClass import TrainingData

# Read the original image


# get file directory
# get all image names within the directory
# create a method which returns a list of images from one main image
# create a method which loops through all the images within the file to obtain a final list
# do machine learning things
# use one of the test images to see whether it works

# step by step: get all images in files, get the size of that list, multiply by 0.8 rounded up,
# for loop in range of the rounded value, pick a random image, remove that image from the original list
# add the remaining to the query list.

# go through each training list, generate the values, put into the arff file.
# go through each query list, generate the values, put into arff.

training_list = []
training_list.append(TrainingData("C:/Users/Katch/Desktop/grain/wholegrain/", "wholegrain"))
training_list.append(TrainingData("C:/Users/Katch/Desktop/grain/groat1/", "groats"))
training_list.append(TrainingData("C:/Users/Katch/Desktop/grain/groat2/", "groats"))
training_list.append(TrainingData("C:/Users/Katch/Desktop/grain/groat3/", "groats"))
training_list.append(TrainingData("C:/Users/Katch/Desktop/grain/broken/", "broken"))

# training_list.append(TrainingData("C:/Users/Katch/Desktop/grain/testing/", "?"))

# get a list of all the images within the files into the objects
# get 80% of the images at random to put into training set, and 20% into testing
for file in training_list:
    images = []
    for item in os.listdir(file.file_dir):
        if item.endswith(".tif") | item.endswith(".jpg"):
            images.append(item)
    file.image_list = images
    file.list_size = len(file.image_list)
    file.training_size = round(file.list_size * 0.8)
    #print(file.list_size)
    #print(file.training_size)
    #print(file.image_list)
    file.query_size = file.list_size - file.training_size
    for x in range(file.training_size):
        #print(len(file.image_list))
        rand_img = file.image_list[randint(0, len(file.image_list)-1)]
        #print(rand_img)
        file.image_list.remove(rand_img)
        file.training_list.append(rand_img)
        #print(file.training_list)
    file.query_list.extend(file.image_list)
    #print(file.training_list)
    #print(file.query_list)
    #print("-------------")



# for each file, get a list of colours, area and then make a list of the grain type based on the list size

def generateTrainingData(index):
    #createARFF("TrainingData/training_dataTemp.arff")

    print("starting file")
    for image in training_list[index].training_list:
        process_image("TrainingData/training_dataTemp.arff", image, training_list[index].file_dir, training_list[index].grain_type, False)
    #generateQueryForImage()


def generateQueryForImage():
    createARFF("TrainingData/query.arff")
    for x in range(5):
        for image in training_list[x].query_list:
            process_image("TrainingData/query.arff", image, training_list[x].file_dir, training_list[x].grain_type, False)
            displayImage(training_list[x].file_dir, image, getMachineLearningPredictions())

def getMachineLearningPredictions():
    return machineLearningAlgorithm()


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

if __name__ =="__main__":
    createARFF("TrainingData/training_dataTemp.arff")

    t1 = threading.Thread(target=generateTrainingData, args=(0,))
    t2 = threading.Thread(target=generateTrainingData, args=(1,))
    t3 = threading.Thread(target=generateTrainingData, args=(2,))
    t4 = threading.Thread(target=generateTrainingData, args=(3,))
    t5 = threading.Thread(target=generateTrainingData, args=(4,))

    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()
    generateQueryForImage()

# step by step for part 1 of demo
# get image from the testing file, extract the grains from the image.
# for each grain get the values and parse them through the machine learning model
# return list of values to opencv, for every grain display what it was detected as next to it
#