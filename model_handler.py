# -*- coding: utf-8 -*-
import random
import keras
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import gc
"""
from keras.models import Sequential
from keras.utils import np_utils

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras.callbacks import LearningRateScheduler 
"""
import numpy as np
import csv 
#import matplotlib as plt
import cv2

PATH_TRAIN = "trainset.csv"
PATH_TEST = "testset.csv"
PATH_SAMPLE = "samplesubmission.csv" 

def row_to_matrix_train(list1,amount_images):
    result = []
    labels = []
    i = 0
    for row in list1:
        labels.append(float(row[0]))
        del row[0]
        row = map(float,row)
        row = list(row)
        #result.append(cv2.resize(row,(28,28),interpolation= 1))
        matrix = np.asmatrix(row)
        matrix= matrix.reshape(28,28) #X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        result.append(matrix)
        i +=1
        if i >= amount_images:
            break
    

    result = np.asarray(result)
    result = result.reshape(result.shape[0], 28, 28, 1)

    labels = np.asarray(labels)

    return result,labels    




def row_to_matrix_test(list1,amount_images):
    result = []

    i = 0
    for row in list1:
        row = map(float,row)
        row = list(row)
        matrix = np.asmatrix(row)
        matrix= matrix.reshape(28,28) #X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

        result.append(matrix)
        i +=1
        if i >= amount_images:
            break
    

    result = np.asarray(result)
    result = result.reshape(result.shape[0], 28, 28, 1)

    return result


def open_csv(PATH):
    reader = csv.reader(open(PATH, "rt",encoding= 'utf-8'), delimiter=",")
    x = list(reader)
    del x[0] #removes labels_name,pixels name row
    del (reader)
    gc.collect()
    return x


def show_image(images,i):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    plt.imshow(images[i], cmap='gray', vmin=0, vmax=255)
    plt.show()

#train_list = open_csv(PATH_TRAIN)
#train_pixels,labels = row_to_matrix_train(train_list,200)
test_list = open_csv(PATH_TEST)
test_pixels = row_to_matrix_test(test_list,200)
from keras.models import load_model 
model = load_model('group_24.h5')
model.load_weights("model_weights.h5")
CLASSES = [0,1,2,3,4,5,6,7,8,9]
#eva=model.evaluate(train_pixels,labels,32)

pred = model.predict(test_pixels[:3])
print(pred)
"""
for i in range(2):
    
    print(pred[i])
    show_image(test_pixels,i)
""" 