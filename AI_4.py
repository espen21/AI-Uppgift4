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
import gc
from sklearn.model_selection import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PATH_TRAIN = "trainset.csv"
PATH_TEST = "testset.csv"
PATH_SAMPLE = "samplesubmission.csv" 

BATCH_SiZE = 32
NUM_CLASSES = 10

#Input: in an matrix of img_matrices, i = amount of images
def show_image(images,i):

    plt.imshow(images[i], cmap='gray', vmin=0, vmax=255)
    plt.show()

def lrs(epoch):
    learn_rate = 0.001
    if epoch > 75:
        learn_rate = 0.0005
    
    return learn_rate

def row_to_matrix_train(list1,amount_image):
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
        if i >= amount_image:
            break
    

    result = np.asarray(result)
    labels = np.asarray(labels)
    result = result.reshape(result.shape[0], 28, 28, 1)

    return result,labels    


def open_csv(PATH):
    reader = csv.reader(open(PATH, "rt",encoding= 'utf-8'), delimiter=",")
    x = list(reader)
    del x[0] #removes labels_name,pixels name row
    del (reader)
    gc.collect()
    return x


def normalize_matrix(pixel_matrix):
    pixel_matrix = pixel_matrix.astype('float32')
    mean = np.mean(pixel_matrix)
    std = np.std(pixel_matrix)
    pixel_matrix = (pixel_matrix-mean)/(std+1e-7)
    return pixel_matrix
       
train_list = open_csv(PATH_TRAIN)
train_pixels,train_labels = row_to_matrix_train(train_list,30000)

train_pixels = normalize_matrix(train_pixels)
train_labels = np_utils.to_categorical(train_labels,NUM_CLASSES)

X_train, X_Val,y_train ,y_val = train_test_split(train_pixels,train_labels,test_size = 0.20,random_state = 2) 




weight_decay = 1e-4



model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
 

model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
 

model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))
 
model.add(Conv2D(512, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(512, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))


model.add(Flatten())
model.add(Dense(NUM_CLASSES, activation='softmax'))
 
model.summary()
callback = keras.callbacks.LearningRateScheduler(lrs)

train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    )

train_datagen.fit(X_train)

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(lr=0.001,decay=1e-6), metrics=['accuracy'])
          




model.fit(train_datagen.flow(X_train, y_train, batch_size=BATCH_SiZE),\
                    steps_per_epoch=X_train.shape[0] // BATCH_SiZE,epochs=64,\
                    verbose=1,validation_data=(X_Val,y_val),callbacks=[callback])


model.save("group_24.h5")
model.save_weights('model_weights.h5')


