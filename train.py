from unet_model import *
import os
from sklearn.model_selection import train_test_split
import numpy as np
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

path = os.getcwd()

weights_path = "./weight.h5"
N_BANDS = 16
N_CLASSES = 3  
CLASS_WEIGHTS = [0.5,0.3,0.2]
N_EPOCHS = 150
UPCONV = True
PATCH_SZ = 288   # should divide by 16
BATCH_SIZE = 150
TRAIN_SZ = 4000  # train size
VAL_SZ = 1000    # validation size

def get_model():
    return unet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS, upconv=UPCONV, class_weights=CLASS_WEIGHTS)
model = get_model()
model.summary()

X = np.load(os.path.join(path,'Train_data.npy'))
y = np.load('./label.npy')
x_train, x_val, y_train, y_val = train_test_split(
    X, y, test_size=0.33, random_state=42)

model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
csv_logger = CSVLogger('log_unet.csv', append=True, separator=';')
tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True, write_images=True)

model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                  verbose=2, shuffle=True,
                  callbacks=[model_checkpoint, csv_logger, tensorboard],
                  validation_data=(x_val, y_val))