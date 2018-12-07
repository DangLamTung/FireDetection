import glob
import os
import cv2
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

path = os.getcwd()
#Image path

folderPath = os.path.join(path, 'Captured')
folderList = os.listdir(folderPath)

def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x

#Sort input by number
folderList = list(map(int, folderList))
folderList.sort()
folderList = list(map(str, folderList))

train_data = []
temp = []
for x in folderList:
    filePath = x
    print(x)
    for i in os.listdir(folderPath + "//" + str(x)):
        subFolderPath = folderPath + "//" + str(x) + "//" + str(i)
        dem = 0 
        temp = []  
        for filename in glob.glob(os.path.join(subFolderPath, '*.csv')):
            if(filename.find("data")!=-1):
                temp1 = normalize(genfromtxt(filename, delimiter=',')[6:294,0:288])
                temp.append(temp1)
        temp_np = np.array(temp)
        temp_np = np.reshape(temp_np,(288,288,16))
        train_data.append(temp_np)
train_data = np.array(train_data)
print(train_data.shape)

np.save("Train_data.npy", train_data)




