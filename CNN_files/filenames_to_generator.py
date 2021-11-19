import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid import make_axes_locatable
import numpy as np
import imageio
import cv2 as cv
import os
from tqdm import tqdm
import random
import h5py


import os
import shutil
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def get_filenames(path_to_dataset, test_size, random_state):
    dir = path_to_dataset
    X_dir = fr'{dir}\resc_wrpd'
    y_dir = fr'{dir}\resc_wrpd_count'
    print(X_dir)
    print(y_dir)
    #y_dir = r'D:\Datasets\dataset_3_Comb_all_max24pi_256x256_11-16-2021_14-07-29\resc_wrpd_count'


    filenames_counter = 0
    X_filenames = []
    y_filenames = []
    for subdir, dirs, files in os.walk(X_dir):
        #print(files)
        for file in files:
            X_filenames.append(file)


    print(len(X_filenames))

    for subdir, dirs, files in os.walk(y_dir):
        # print(files)
        for file in files:
            y_filenames.append(file)



    print(len(y_filenames))

    print(X_filenames[:10])
    print(y_filenames[:10])

    X_filenames_shuffled, y_filenames_shuffled = shuffle(X_filenames, y_filenames)

    np.save('X_files_shuf.npy', X_filenames_shuffled)
    np.save('y_files_shuf.npy', y_filenames_shuffled)

    X_train_filenames, X_val_filenames, y_train_filenames, y_val_filenames = train_test_split(
        np.array(X_filenames_shuffled), np.array(y_filenames_shuffled), test_size=test_size, random_state=random_state)

    return X_train_filenames, X_val_filenames, y_train_filenames, y_val_filenames
