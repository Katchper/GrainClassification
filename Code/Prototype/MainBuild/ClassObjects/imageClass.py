class ImageData:
    def __init__(self, file_dir, image_name, grain_type, is_training, range_val):
        self.file_dir = file_dir
        self.image_name = image_name
        self.grain_type = grain_type
        self.is_training = is_training
        self.range_val = range_val