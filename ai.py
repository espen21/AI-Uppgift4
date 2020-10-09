# -*- coding: utf-8 -*-

import sys
import os 
import random
import gc 
import numpy as np  
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg

PATH_TRAIN = "trainset.csv"
PATH_TEST = "testset.csv"
PATH_SAMPLE = "samplesubmission.csv" 

print("Hej")


from keras.datasets import cifar10
def show_imgs(X):
    plt.figure(1)
    k = 0
    for i in range(0,4):
        for j in range(0,4):
            plt.subplot2grid((4,4),(i,j))
            plt.imshow(toimage(X[k]))
            k = k+1
    # show the plot
    plt.show()
 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
show_imgs(x_test[:16])