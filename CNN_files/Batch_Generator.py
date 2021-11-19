import numpy as np

from skimage.transform import resize
from skimage.io import imread
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import img_to_array, load_img

class Batch_Generator(Sequence):
    def __init__(self, X_filenames, y_filenames , batch_size, dir):
        self.X_filenames = X_filenames
        self.y_filenames = y_filenames
        self.batch_size = batch_size
        self.dir = dir

    def __len__(self):
        return (np.ceil(len(self.X_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.X_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.y_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]

        return np.array([
            resize(img_to_array(load_img(self.dir + "/resc_wrpd/" + str(file_name))).astype('float32'), (256,256,1))
            for file_name in batch_x]), np.array([
            np.load(self.dir + "/resc_wrpd_count/" + str(file_name)).astype(np.uint8)
            for file_name in batch_y])