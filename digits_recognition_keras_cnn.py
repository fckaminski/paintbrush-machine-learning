#Jason Brownlee, Machine Learning Algorithms in Python, Machine Learning Mastery,
#Available from https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/

#define wehther to use CPU or GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# 1. Import modules, classes and functions
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import Sequential
from pathlib import Path
from keras.models import load_model
import math
from keras.datasets import mnist
from keras.utils.data_utils import get_file
import numpy as np
import time

''' load from offline 'mnist.npz'
def load_data(path='mnist.npz'):
   
    with np.load('mnist.npz', allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)'''

def digits_recognition():

    FILE_NAME = "digits_recognition_cnn.h5"
   
    # 2. Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    num_classes = 10   #possible results: numbers from 0 to 9
    image_bit_length = 28

    #the layers used for two-dimensional convolutions expect pixel values with the dimensions [pixels][width][height][channels]
    #In MNIST where the pixel values are gray scale, the pixel dimension is set to 1. So shape will be: (samples x 28 x 28 x 1)
    x_train = x_train.reshape(x_train.shape[0], image_bit_length, image_bit_length, 1)  
    x_test = x_test.reshape(x_test.shape[0], image_bit_length, image_bit_length, 1)
    input_shape = (image_bit_length, image_bit_length, 1)
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    

    model_file = Path(FILE_NAME)
    if not model_file.is_file():

        # 4. Create the classification model and train (fit) 
        model = Sequential()
        model.add(Conv2D(30, kernel_size=(5, 5),activation='relu',input_shape=input_shape))
        model.add(MaxPooling2D())
        model.add(Conv2D(15, (3, 3), activation='relu'))   
        model.add(MaxPooling2D( pool_size=(2, 2)  ))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

        print("Training model...\n")        
        # Fit (train) the classification model
        start_time = time.time()
        model.fit(x_train, y_train, epochs=12, batch_size=128, verbose=0)  #100 10
        elapsed_time = time.time() - start_time
        print("Calculation time: ", "%0.2f" % elapsed_time, "s")
        # save model and architecture to single file
        model.save(FILE_NAME)
        
    else:
        print("Loading saved model...\n")
        model = load_model(FILE_NAME)

    model.summary()
    
    # 5. Test the classification model
    result = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss: %.2f%%" % (result[0]*100) )
    print("Test accuracy: %.2f%%" % (result[1]*100))

    return model

if __name__ == '__main__':
    digits_recognition()
    
