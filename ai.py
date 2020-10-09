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

BATCH_SiZE = 32


def show_image(images,i):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    plt.imshow(images[i], cmap='gray', vmin=0, vmax=255)
    plt.show()

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
        labels.append(float(row[0]))
        del row[0]
        row = map(float,row)
        row = list(row)
        #result.append(cv2.resize(row,(28,28),interpolation= 1))
        matrix = np.asmatrix(row)
        matrix= matrix.reshape(28,28)
        result.append(matrix)
        i +=1
        if i >= amount_image:
            break
    

    result = np.asarray(result)
    labels = np.asarray(labels)

    return result,labels    


def open_csv(PATH):
    reader = csv.reader(open(PATH, "rt",encoding= 'utf-8'), delimiter=",")
    x = list(reader)
    del x[0] #removes labels_name,pixels name row
    del (reader)
    gc.collect()
    return x

       
train_list = open_csv(PATH_TRAIN)
train_pixels,train_labels = row_to_matrix_train(train_list,2000)
#print(train_labels)

#test_list = open_csv(PATH_TEST)
#test_pixels,test_labels = row_to_matrix(test_list)

#mean = np.mean(train_pixels,axis=(0,1,2,3))
#std = np.std(train_pixels,axis=(0,1,2,3))
#train_pixels = (train_pixels-mean)/(std+1e-7)
#x_test = (x_test-mean)/(std+1e-7)
 
print(len(train_pixels))
print(len(train_labels))
from sklearn.model_selection import train_test_split
X_train, X_Val,y_train ,y_val = train_test_split(train_pixels,train_labels,test_size = 0.20,random_state = 2) 
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_Val = X_Val.reshape(X_Val.shape[0], 28, 28, 1)

print(y_train)
print(X_train.shape)
print(y_train.shape)
"""
model = models.Sequential()
model.add(layers.Conv2D(32,(1,1),activation ='elu',input_shape =(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64,(1,1),activation ='elu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128,(1,1),activation ='elu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128,(1,1),activation ='elu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dropout(0.3))
model.add(layers.Dense(512,activation='elu'))
model.add(layers.Dense(1,activation='softmax'))
model.summary()
"""
import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler
weight_decay = 1e-4
num_classes = 10

y_train = np_utils.to_categorical(y_train,num_classes)
y_val = np_utils.to_categorical(y_val,num_classes)
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=X_train.shape[1:]))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
 
model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
 
model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))
 
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))
 
model.summary()
 
model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.001,decay=1e-6), metrics=['accuracy'])
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range= 0.2,
                                   height_shift_range= 0.2,
                                   shear_range= 0.2,
                                   zoom_range= 0.2,
                                   horizontal_flip= True,)             
                                                
                                                
val_datagen = ImageDataGenerator(rescale=1./255)
train_gen = train_datagen.flow(X_train,y_train,batch_size= BATCH_SiZE)
val_gen = val_datagen.flow(X_Val,y_val,batch_size=BATCH_SiZE)

ntrain = len(X_train)
nval = len(X_Val)
history = model.fit_generator(train_gen,
                              steps_per_epoch = ntrain //BATCH_SiZE,
                              epochs= 64,
                              validation_data= val_gen,
                              validation_steps=nval//BATCH_SiZE)

model.save_weights('model_weights.h5')
model.save("group_24.h5")

