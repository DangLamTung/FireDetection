from unet_model import *
N_BANDS = 16
N_CLASSES = 3  # buildings, roads, trees, crops and water
CLASS_WEIGHTS = [0.2, 0.3, 0.1, 0.1, 0.3]
N_EPOCHS = 150
UPCONV = True
PATCH_SZ = 288   # should divide by 16
BATCH_SIZE = 150
TRAIN_SZ = 4000  # train size
VAL_SZ = 1000    # validation size


def get_model():
    return unet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS, upconv=UPCONV, class_weights=[0.5,0.1,0.4])
model = get_model()
model.summary()