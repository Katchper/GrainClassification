import time
from multiprocessing import Pool
from random import randint

import cv2
import os
import threading

import numpy as np

from Code.Prototype.MainBuild.ClassObjects.imageClass import ImageData
from Code.Prototype.MainBuild.FileMethods.arffBuilder import createARFF, writeLineToARFF, writeArffFromArray
from Code.Prototype.MainBuild.ImageMethods.huMoments import process_image, process_image_to_values
from Code.Prototype.MainBuild.imageProcessing import read_image, displayImage
from Code.Prototype.MainBuild.machineLearning import machineLearningAlgorithm, startJvm, stopJvm
from Code.Prototype.MainBuild.trainingDataClass import TrainingData

# for each file, get a list of colours, area and then make a list of the grain type based on the list size

def generateTrainingData():
    createARFF("FileMethods/TrainingData/training_dataTemp.arff")
    for x in range(5):
        print("starting file")
        for image in training_list[x].training_list:
            process_image("FileMethods/TrainingData/training_dataTemp.arff", image, training_list[x].file_dir, training_list[x].grain_type, False)
    generateQueryForImage()


def generateTrainingLists(training_item):
    hue_list = []
    sat_list = []
    val_list = []
    area_list = []
    perimeter_list = []
    hu_moments_list = []
    grain_name = []
    for image in training_item.query_list:
        print("Starting image")
        hue, sat, val, area, perimeter, hu, grain = process_image_to_values(image, training_item.file_dir, training_item.grain_type)
        hue_list.extend(hue)
        sat_list.extend(sat)
        val_list.extend(val)
        area_list.extend(area)
        perimeter_list.extend(perimeter)
        hu_moments_list.extend(hu)
        grain_name.extend(grain)

    return hue_list, sat_list, val_list, area_list, perimeter_list, hu_moments_list, grain_name

def generateListForImage(image):
    #print("starting image")
    return process_image_to_values(image.image_name, image.file_dir, image.grain_type)

def generateQueryForImage():
    createARFF("FileMethods/TrainingData/query.arff")
    for x in range(5):
        for image in training_list[x].query_list:
            process_image("FileMethods/TrainingData/query.arff", image, training_list[x].file_dir, training_list[x].grain_type, False)
            displayImage(training_list[x].file_dir, image, getMachineLearningPredictions("TrainingData/training_data.arff", "TrainingData/query.arff"))

def visualiseMachineLearning(training_data, query_data, x, image_name, training_list):
    createARFF(query_data)
    process_image(query_data, image_name, training_list[x].file_dir, training_list[x].grain_type, False)
    displayImage(training_list[x].file_dir, image_name, getMachineLearningPredictions(training_data, query_data))


def getMachineLearningPredictions(training, query):
    return machineLearningAlgorithm(training, query)


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
    startJvm()
    for x in range(20):
        tic = time.perf_counter()
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
            # print(file.list_size)
            # print(file.training_size)
            # print(file.image_list)
            file.query_size = file.list_size - file.training_size
            for x in range(file.training_size):
                # print(len(file.image_list))
                rand_img = file.image_list[randint(0, len(file.image_list) - 1)]
                # print(rand_img)
                file.image_list.remove(rand_img)
                file.training_list.append(rand_img)
                # print(file.training_list)
            file.query_list.extend(file.image_list)
            # print(file.training_list)
            # print(file.query_list)
            # print("-------------")

        training_images = []
        for item in training_list:
            for image in item.training_list:
                training_images.append(ImageData(item.file_dir, image, item.grain_type))

        query_images = []
        for item in training_list:
            for image in item.query_list:
                query_images.append(ImageData(item.file_dir, image, item.grain_type))

        with Pool() as pool:
            results = pool.map(generateListForImage, training_images)
            hue_temp, sat_temp, val_temp, area_temp, perimeter_temp, hu_temp, grain_temp = zip(*results)
            hue_final = np.concatenate(hue_temp)
            sat_final = np.concatenate(sat_temp)
            val_final = np.concatenate(val_temp)
            area_final = np.concatenate(area_temp)
            perimeter_final = np.concatenate(perimeter_temp)
            hu_final = np.concatenate(hu_temp)
            grain_final = np.concatenate(grain_temp)
            writeArffFromArray("FileMethods/TrainingData/training_dataTemp.arff", hue_final, sat_final, val_final, area_final, perimeter_final, hu_final, grain_final)

        with Pool() as pool:
            results = pool.map(generateListForImage, query_images)
            hue_temp, sat_temp, val_temp, area_temp, perimeter_temp, hu_temp, grain_temp = zip(*results)
            hue_final = np.concatenate(hue_temp)
            sat_final = np.concatenate(sat_temp)
            val_final = np.concatenate(val_temp)
            area_final = np.concatenate(area_temp)
            perimeter_final = np.concatenate(perimeter_temp)
            hu_final = np.concatenate(hu_temp)
            grain_final = np.concatenate(grain_temp)
            writeArffFromArray("FileMethods/TrainingData/query_dataTemp.arff", hue_final, sat_final, val_final, area_final, perimeter_final, hu_final, grain_final)

        getMachineLearningPredictions("TrainingData/training_dataTemp.arff", "TrainingData/query_dataTemp.arff")
        toc = time.perf_counter()
        print(f"completed iteration in {toc - tic:0.4f} seconds")
    stopJvm()
# step by step for part 1 of demo
# get image from the testing file, extract the grains from the image.
# for each grain get the values and parse them through the machine learning model
# return list of values to opencv, for every grain display what it was detected as next to it
#