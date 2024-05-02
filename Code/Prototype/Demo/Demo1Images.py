"""
Demo 1
Author: Kacper Dziedzic ktd1
Version: 1.1

Showing the image preprocessing techniques being used for the project.
This includes showing the greyscale conversion, applying a threshold and getting contours.
"""


import os
from random import randint
from Code.Prototype.MainBuild.ClassObjects.imageClass import ImageData
from Code.Prototype.MainBuild.ImageMethods.Main.imagePreprocessing import process_image_demo
from Code.Prototype.MainBuild.ClassObjects.trainingDataClass import TrainingData

training_list = []
training_list.append(TrainingData("C:/Users/Katch/Desktop/grain/wholegrain/", "wholegrain", 0.1, 0.1))
training_list.append(TrainingData("C:/Users/Katch/Desktop/grain/groat/", "groats", 0.1, 0.1))
training_list.append(TrainingData("C:/Users/Katch/Desktop/grain/broken/", "broken", 0.1, 0.1))
training_list.append(TrainingData("C:/Users/Katch/Desktop/grain/broken2/", "broken", 0.1, 0.1))

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
        training_images.append(ImageData(item.file_dir, image, item.grain_type, False, 1))

query_images = []
for item in training_list:
    for image in item.query_list:
        query_images.append(ImageData(item.file_dir, image, item.grain_type, False, 0))


grainType = 1

print(training_list[grainType].file_dir)

process_image_demo(training_list[grainType].image_list[1], training_list[grainType].file_dir, training_list[grainType].grain_type)