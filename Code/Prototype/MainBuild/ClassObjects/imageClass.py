"""
ImageData class object
Author: Kacper Dziedzic ktd1
Version: 1.1

dedicated to storing all relevant information associated with an Image
file_dir = the directory pointing to the image
image_name = the name of the image within the folder
grain_type = the category of grain of the image, should be the name of the parent folder.
is_training = value saying whether this image is part of the training data set or not
range_val = variable storing the current iteration of the program for storing

"""


class ImageData:
    def __init__(self, file_dir, image_name, grain_type, is_training, range_val):
        self.file_dir = file_dir
        self.image_name = image_name
        self.grain_type = grain_type
        self.is_training = is_training
        self.range_val = range_val