"""
TrainingData class object
Author: Kacper Dziedzic ktd1
Version: 1.1

Class dedicated to storing all the data associated with a particular training data file within file storage.
image_list = stores an array of all the images within the folder
file_dir = the directory to the folder
grain_type = the grain type associated with the folder
training_list = the list of images to be used for the training data
query_list = the list of images to be used for the query data
list_size = the total amount of images within the folder
training_size = the amount of images to be used for creating the training_list
query_size = the amount of images to be used for creating the query_list
"""

class TrainingData:
    def __init__(self, file_dir, grain_type, training_percent, query_percent):
        self.image_list = None
        self.file_dir = file_dir
        self.grain_type = grain_type
        self.training_list = []
        self.query_list = []
        self.training_percent = training_percent
        self.query_percent = query_percent
        self.list_size = 0
        self.training_size = 0
        self.query_size = 0