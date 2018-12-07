import glob
import os
import cv2
import numpy as np
from numpy import genfromtxt

path = os.getcwd()
#Image path


label = []

#Label path
labelPath = os.path.join(path, 'Label')
labelList = os.listdir(labelPath)

for filename in glob.glob(os.path.join(labelPath, '*.csv')):
    true_label = []
    temp = genfromtxt(filename, delimiter=',')[6:294,0:288]
    print(temp.shape)
    temp_label1 = np.zeros((288,288))
    temp_label1[np.where(temp==0)] = 1

    temp_label2 = np.zeros((288,288))
    temp_label2[np.where(temp==0)] = 0

    temp_label3 = np.zeros((288,288))
    temp_label3[np.where(temp==0)] = -1

    true_label.append(temp_label1)
    true_label.append(temp_label2)
    true_label.append(temp_label3)

    label.append(true_label)
label = np.array(label)
print(label.shape)
np.save("label.npy", label)
