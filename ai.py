# -*- coding: utf-8 -*-

import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras.callbacks import LearningRateScheduler 
import numpy as np
import csv 
#import matplotlib as plt
import cv2
PATH_TRAIN = "trainset.csv"
PATH_TEST = "testset.csv"
PATH_SAMPLE = "samplesubmission.csv" 

print("Hej")

WIDTH =  14
HEIGHT = 56
def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    if epoch > 100:
        lrate = 0.0003
    return lrate

def row_to_matrix_train(list1,amount_image):
    result = []
    labels = []
    i = 0
    for row in list1:
        labels.append(row[0])
        del row[0]
        row = map(float,row)
        row = list(row)
        matrix = np.asmatrix(row)
        matrix= matrix.reshape(28,28)
        result.append(matrix)
        i +=1
        if i > amount_image:
            break
    return result,labels    


def open_csv(PATH):
    reader = csv.reader(open(PATH, "rt",encoding= 'utf-8'), delimiter=",")
    x = list(reader)
    del x[0] #removes labels_name,pixels name row
    return x

       
train_list = open_csv(PATH_TRAIN)
train_pixels,train_labels = row_to_matrix_train(train_list,10)
print(type(train_pixels))
#print(train_labels)

#test_list = open_csv(PATH_TEST)
#test_pixels,test_labels = row_to_matrix(test_list)

#mean = np.mean(train_pixels,axis=(0,1,2,3))
#std = np.std(train_pixels,axis=(0,1,2,3))
#train_pixels = (train_pixels-mean)/(std+1e-7)
#x_test = (x_test-mean)/(std+1e-7)
 
from PIL import Image
#print((train_pixels[0]))
print(train_pixels[0])
print(train_labels)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.imshow(train_pixels[0], cmap='gray', vmin=0, vmax=255)
plt.show()
