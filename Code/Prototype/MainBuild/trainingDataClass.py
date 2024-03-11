class TrainingData:
    def __init__(self, file_dir, grain_type):
        self.image_list = None
        self.file_dir = file_dir
        self.grain_type = grain_type
        self.training_list = []
        self.query_list = []
        self.list_size = 0
        self.training_size = 0
        self.query_size = 0

        def getFile(self):
            return self.file_dir

        def getGrain(self):
            return self.file_dir
