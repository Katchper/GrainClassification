import os
import cv2
from random import randint
from Code.Prototype.MainBuild.ClassObjects.imageClass import ImageData
from Code.Prototype.MainBuild.ImageMethods.huMoments import process_image_demo
from Code.Prototype.MainBuild.machineLearning import startJvm
from Code.Prototype.MainBuild.main import visualiseMachineLearning
from Code.Prototype.MainBuild.trainingDataClass import TrainingData

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

training_list[0].file_dir
process_image_demo(training_list[0].image_list[0], training_list[0].file_dir)