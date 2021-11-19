import json
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from datetime import datetime
from tqdm import tqdm
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard,CSVLogger,LambdaCallback
from keras.layers import Input
from keras.losses import MeanSquaredError, SparseCategoricalCrossentropy
from keras.preprocessing.image import img_to_array, load_img
from keras.utils.vis_utils import plot_model
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

from CNNs import UNetResNet_5lvl,UNetResNet_5lvl_Expanded, UNetResNet_4lvl, SimpleNet_20_10, UNetResNet_5lvl_softmax, UNetResNet_5lvl_softmax_withReg
from CNNs import cleanup,genConfig,plotter, Batch_Generator, filenames_to_generator


print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_gpu_available(cuda_only=True))

date = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

### DEFINITION OF HYPERPARAMETERS
test_size = 0.2
random_state = 18
n_filters = 4
stop_patience = 8
batch_size = 32
epoch_limit = 200

### DEFINITION OF DIRECTORY FOR THE MODEL
cleanup.clearEmptyDirectories(r"C:\Users\Michał\PycharmProjects\BadaniaMchtr\CNNs\Results")
pathtoDataSets = "D:/Datasets"
dataset = "dataset_3_Comb_all_max24pi_256x256_11-16-2021_14-07-29"
pathtoExactDataset = f"{pathtoDataSets}/{dataset}"
type1 = "resc_wrpd"
type2 = "resc_wrpd_count"
comment = "SparseCategoricalCrossEntropy_BatchGen"

path_to_inputs = os.path.join(pathtoDataSets,dataset).replace("\\", "/")
path_to_outputs = os.path.join(pathtoDataSets,dataset).replace("\\", "/")
print(f"Input path: {path_to_inputs}")
print(f"Output path: {path_to_outputs}")

### CUSTOM JSON CALLBACK





def SSIMLoss(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 255.0))

def SIMM_MSE(y_true, y_pred):
  return MeanSquaredError - SSIMLoss(y_true, y_pred)
### IMPORTING IMAGES TO PROGRAM





y = np.load(r"C:\dataset3\w_dataset3.npy").astype(np.uint8)

num_classes = int((np.amax(y)-np.amin(y)) +1)
del y
print(num_classes)

name = f"UnetResnet_{type1}_{date}_{comment}_{num_classes}"


os.makedirs('Results/' + name, exist_ok=True)
os.makedirs('Results/' + name+'/model', exist_ok=True)
os.makedirs('Results/' + name+'/cfg', exist_ok=True)
cfg_dir = 'Results/' + name +'/cfg'
json_log = open(f'Results/{name}/{name}loss_log.json', mode='wt', buffering=1)
json_logging_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: json_log.write(
        json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n'),
    on_train_end=lambda logs: json_log.close()
)

genConfig.write_cfg(cfg_dir, name, 'w', type_input = type1, type_output= type2, dataset = dataset)
### SPLITTING THE SETS
X_train_filenames, X_val_filenames, y_train_filenames, y_val_filenames = filenames_to_generator.get_filenames(pathtoExactDataset, test_size=test_size, random_state=random_state)


my_training_batch_generator = Batch_Generator.Batch_Generator(X_train_filenames, y_train_filenames, batch_size, pathtoExactDataset)
my_validation_batch_generator = Batch_Generator.Batch_Generator(X_val_filenames, y_val_filenames, batch_size, pathtoExactDataset)
### DEFINING CALLBACKS

callbacks = [
    EarlyStopping(patience=stop_patience, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000001, verbose=1),
    ModelCheckpoint(f'Results/{name}/model/{name}.h5', verbose=1, save_best_only=True),
    #Tensordash(ModelName=f"{name}", email='mgontarz15@gmail.com', password='dupadupa'),
    TensorBoard(log_dir=f"Results/{name}/logs", write_graph=True, write_images= True, update_freq=5),
    CSVLogger(f"Results/{name}/{name}.csv"),
    json_logging_callback,
            ]
genConfig.write_cfg(cfg_dir, name, "a", test_size = test_size, random_state = random_state, classes = num_classes)


### INITIALIZING MODEL
input_img = Input((256, 256, 1), name='img')
model = UNetResNet_5lvl_softmax_withReg.get_unet(input_img, num_classes, n_filters=n_filters, kernel_regularizer = 'l2')
#model = keras.models.load_model("C:/Users/Michał/PycharmProjects/BadaniaMchtr/CNNs/Results/UNetResNet5lvl_resc_wrpd_08-21-2021_18-11-20/model/UNetResNet5lvl_resc_wrpd_08-21-2021_18-11-20.h5")
model.compile(optimizer=Adam(learning_rate=0.0001), loss=SparseCategoricalCrossentropy(),  metrics=['accuracy'])
model.summary()
start = datetime.now()
genConfig.write_cfg(cfg_dir,name, 'a', optimizer = "Adam", loss = comment, metrics = "accuracy", state = "new", n_filters = n_filters)

### MODEL TRAINING

results = model.fit_generator(generator=my_training_batch_generator, steps_per_epoch=int(16000//batch_size), epochs= epoch_limit, validation_data=my_validation_batch_generator, validation_steps=int(4000//batch_size), callbacks = callbacks)

stop = datetime.now()

time = stop-start

genConfig.write_cfg(cfg_dir,name, "a", batch_size = batch_size, epoch_limit = epoch_limit, time = time)

### PLOTTING RESULTS
plotter.plot_model_data(model, name, results)
genConfig.write_cfg(cfg_dir,name, "a",  best_loss = np.min(results.history["val_loss"]), for_epoch_loss = np.argmin(results.history["val_loss"]), best_acc = np.max(results.history["val_accuracy"]), for_epoch_acc = np.argmax(results.history["val_accuracy"]))


### EVALUATING MODEL - SAVING CFG
score = model.evaluate_generator(my_training_batch_generator, steps = int(5000//batch_size))
labels = model.metrics_names

header_train  = "\n-------TRAIN EVALUATION-------"
loss_list = []


for i, loss in enumerate(labels):
    loss_list.append(f"| {loss} : {score[i]} |")

genConfig.write_cfg(cfg_dir,name, "a", _ = header_train, TRAIN_EVAL = "".join(loss_list))

loss_list.clear()

score = model.evaluate_generator(my_validation_batch_generator, steps = int(5000//batch_size))


header_test = "\n-------TEST EVALUATION-------"

for i, loss in enumerate(labels):
    loss_list.append(f"| {loss} : {score[i]} |")

genConfig.write_cfg(cfg_dir,name, "a", _ = header_train, TEST_EVAL = "".join(loss_list))
