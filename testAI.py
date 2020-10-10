import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import numpy as np
import csv
import gc
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
    result = result.reshape(result.shape[0], 28, 28, 1)

    return result,labels    


def open_csv(PATH):
    reader = csv.reader(open(PATH, "rt",encoding= 'utf-8'), delimiter=",")
    x = list(reader)
    del x[0] #removes labels_name,pixels name row
    del (reader)
    gc.collect()
    return x

       

#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#x_train = x_train[:200]
#y_train = y_train[:200]
PATH_TRAIN = "trainset.csv"
train_list = open_csv(PATH_TRAIN)
x_train,y_train =  row_to_matrix_train(train_list,2000)
#x_test = x_test[:200]
#y_test = y_test[:200]
print(x_train.shape)
x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
mean = np.mean(x_train)
std = np.std(x_train)
x_train = (x_train-mean)/(std+1e-7)


#z-score
#mean = np.mean(x_train,axis=(0,1,2,3))
#std = np.std(x_train,axis=(0,1,2,3))
#x_train = (x_train-mean)/(std+1e-7)
#x_test = (x_test-mean)/(std+1e-7)
print(x_train.shape)
num_classes = 10
y_train = np_utils.to_categorical(y_train,num_classes)
#y_test = np_utils.to_categorical(y_test,num_classes)
from sklearn.model_selection import train_test_split
x_train, x_Val,y_train ,y_val = train_test_split(x_train,y_train,test_size = 0.20,random_state = 2) 
weight_decay = 1e-4

model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
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

#data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    )
datagen.fit(x_train)

#training
batch_size = 64

opt_rms = keras.optimizers.RMSprop(lr=0.001,decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),\
                    steps_per_epoch=x_train.shape[0] // batch_size,epochs=64,\
                    verbose=1,validation_data=(x_Val,y_val),callbacks=[LearningRateScheduler(lr_schedule)])
"""
#save to disk
"""
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

from keras.models import load_model
model.save_weights('test_weights.h5')    
model.save("testAI.h5")
"""
model = load_model('testAI.h5')
pred = model.predict(x_train)
print(pred[:3])
print(y_train[:3])
#testing
#scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
#print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))
"""