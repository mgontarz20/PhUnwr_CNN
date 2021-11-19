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

from CNNs import UNetResNet_5lvl,UNetResNet_5lvl_Expanded, UNetResNet_4lvl, SimpleNet_20_10, UNetResNet_5lvl_softmax
from CNNs import cleanup,genConfig,plotter


print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_gpu_available(cuda_only=True))

date = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

### DEFINITION OF HYPERPARAMETERS
test_size = 0.2
random_state = 17
n_filters = 4
stop_patience = 8
batch_size = 8
epoch_limit = 200
norm = False
coeff = 1.0
if norm:
    coeff = 255.0
activation_function = 'relu'
### DEFINITION OF DIRECTORY FOR THE MODEL
cleanup.clearEmptyDirectories(r"C:\Users\Michał\PycharmProjects\BadaniaMchtr\CNNs\Results")
pathtoDataSet = r"C:/dataset3"
dataset = "dataset3"
type1 = "resc_wrpd"
type2 = "resc_wrpd_count"
comment = "SparseCategoricalCrossEntropy"
name = f"UnetResnet_{type1}_{date}_{comment}"
os.makedirs('Results/' + name, exist_ok=True)
os.makedirs('Results/' + name+'/model', exist_ok=True)
os.makedirs('Results/' + name+'/cfg', exist_ok=True)
cfg_dir = 'Results/' + name +'/cfg'
path_to_inputs = os.path.join(pathtoDataSet,dataset).replace("\\", "/")
path_to_outputs = os.path.join(pathtoDataSet,dataset).replace("\\", "/")
print(f"Input path: {path_to_inputs}")
print(f"Output path: {path_to_outputs}")

### CUSTOM JSON CALLBACK

json_log = open(f'Results/{name}/{name}loss_log.json', mode='wt', buffering=1)
json_logging_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: json_log.write(
        json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n'),
    on_train_end=lambda logs: json_log.close()
)



def SSIMLoss(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 255.0))

def SIMM_MSE(y_true, y_pred):
  return MeanSquaredError - SSIMLoss(y_true, y_pred)
### IMPORTING IMAGES TO PROGRAM


genConfig.write_cfg(cfg_dir, name, 'w', type_input = type1, type_output= type2, dataset = dataset)

f = h5py.File(f"{pathtoDataSet}/dataset3_1.hdf5", 'r')

X_train = f['X_train'][:]
y_train = f['y_train'][:]

X_valid = f['X_valid'][:]
y_valid = f['y_valid'][:]
f.close()
num_classes = int((np.amax(y_train) - np.amin(y_train)) +1)
print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)
print(X_train.dtype, y_train.dtype)
print(X_valid.dtype, y_valid.dtype)
print(num_classes)


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
model = UNetResNet_5lvl_softmax.get_unet(input_img, num_classes, n_filters=n_filters)
#model = keras.models.load_model("C:/Users/Michał/PycharmProjects/BadaniaMchtr/CNNs/Results/UNetResNet5lvl_resc_wrpd_08-21-2021_18-11-20/model/UNetResNet5lvl_resc_wrpd_08-21-2021_18-11-20.h5")
model.compile(optimizer=Adam(learning_rate=0.0001), loss=SparseCategoricalCrossentropy(),  metrics=['accuracy'])
model.summary()
start = datetime.now()
genConfig.write_cfg(cfg_dir,name, 'a', optimizer = "Adam", loss = comment, metrics = "accuracy", state = "new", n_filters = n_filters)

### MODEL TRAINING

results = model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch_limit, callbacks=callbacks,
                     validation_data=(X_valid, y_valid))

stop = datetime.now()

time = stop-start

genConfig.write_cfg(cfg_dir,name, "a", batch_size = batch_size, epoch_limit = epoch_limit, time = time, activation_function = activation_function)

### PLOTTING RESULTS
plotter.plot_model_data(model, name, results)
genConfig.write_cfg(cfg_dir,name, "a",  best_loss = np.min(results.history["val_loss"]), for_epoch_loss = np.argmin(results.history["val_loss"]), best_acc = np.max(results.history["val_accuracy"]), for_epoch_acc = np.argmax(results.history["val_accuracy"]))


### EVALUATING MODEL - SAVING CFG
score = model.evaluate(X_train, y_train,
                            batch_size=batch_size)
labels = model.metrics_names

header_train  = "\n-------TRAIN EVALUATION-------"
loss_list = []


for i, loss in enumerate(labels):
    loss_list.append(f"| {loss} : {score[i]} |")

genConfig.write_cfg(cfg_dir,name, "a", _ = header_train, TRAIN_EVAL = "".join(loss_list))

loss_list.clear()

score= model.evaluate(X_valid, y_valid,
                            batch_size=batch_size)

header_test = "\n-------TEST EVALUATION-------"

for i, loss in enumerate(labels):
    loss_list.append(f"| {loss} : {score[i]} |")

genConfig.write_cfg(cfg_dir,name, "a", _ = header_train, TEST_EVAL = "".join(loss_list))
