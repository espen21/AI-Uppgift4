# -*- coding: utf-8 -*-
import random
import keras
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import gc
import csv
import numpy as np
import cv2
from keras.models import load_model 
from keras.utils import np_utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PATH_TRAIN = "trainset.csv"
PATH_TEST = "testset.csv"
PATH_SAMPLE = "samplesubmission.csv" 

CLASSES = [0,1,2,3,4,5,6,7,8,9]
NUM_CLASSES = 10

def row_to_matrix_train(list1,amount_images):
    result = []
    labels = []
    i = 0
    for row in list1:
        labels.append(float(row[0]))
        del row[0]
        row = map(float,row)
        row = list(row)
        matrix = np.asmatrix(row)
        matrix= matrix.reshape(28,28) 
        result.append(matrix)
        i +=1
        if i >= amount_images  or i >= len(list1):
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
        matrix= matrix.reshape(28,28) 

        result.append(matrix)
        i +=1
        if i >= amount_images or i >= len(list1):
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

    plt.imshow(images[i], cmap='gray', vmin=0, vmax=255)
    plt.show()

def write_submission(pred_list):
    with open('submission.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(("ImageId","Label"))
        img_id = 1
        for value in pred_list:
            
            writer.writerow((str(img_id),str(value)))
            img_id +=1
        print("Done with writing CSV file")

def normalize_matrix(pixel_matrix):
    pixel_matrix = pixel_matrix.astype('float32')
    mean = np.mean(pixel_matrix)
    std = np.std(pixel_matrix)
    pixel_matrix = (pixel_matrix-mean)/(std+1e-7)
    return pixel_matrix

test_list = open_csv(PATH_TEST)
test_pixels = row_to_matrix_test(test_list,300000)



#score 0.96321
model = load_model("group_24.h5")
model.load_weights("model_weights.h5")




x_test = normalize_matrix(test_pixels)
pred = np.argmax(model.predict(x_test),1)
write_submission(pred)
