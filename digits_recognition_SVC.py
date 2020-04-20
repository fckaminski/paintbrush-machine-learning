print(__doc__)

#https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py


# Standard scientific Python imports
import matplotlib.pyplot as plt
from random import randint
from keras.datasets import mnist

import numpy as np
from sklearn import datasets, svm, metrics, calibration
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from pathlib import Path

# ilustra os números do dataset na tela
def plot_number(axes, images_and_labels, line, prelabel):
    for ax, (image, label) in zip(axes[line, :], images_and_labels[:4]):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title('%s: %i' % (prelabel, label) )

def digits_recognition():

    FILE_NAME = "digits_recognition.pickle"
    
    # load dataset MINIST offline
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    #mostra 4 amostras do dataset na tela
    _, axes = plt.subplots(2, 4)
    images_and_labels = list(zip(x_train, y_train))
    plot_number(axes, images_and_labels, line = 0, prelabel = 'Training')
        
    # flatten 28*28 images to a 784 vector for each image
    num_pixels = x_train.shape[1] * x_train.shape[2]
    image_size = x_train.shape[1]
    x_train = x_train.reshape((x_train.shape[0], num_pixels))
    x_test = x_test.reshape((x_test.shape[0], num_pixels))
    x_train = x_train/255.0
    x_test = x_test/255.0 


    model_file = Path(FILE_NAME)
    if not model_file.is_file():

        # Create a classifier: Utilizado Linear SVC porque o SVC apanha para datasets muito grandes
        svmcl = svm.LinearSVC()
        classifier = calibration.CalibratedClassifierCV(svmcl)   
        print("Training model...\n")
        # We learn the digits
      
        classifier.fit(x_train, y_train)

        # save the model to disk
        filename = FILE_NAME
        pickle.dump(classifier, open(filename, 'wb'))      
        
    else:
        print("Loading saved model...\n")
        classifier = pickle.load(open(FILE_NAME, 'rb'))

    # Now predict the value of the digit on the second half:
    predicted = classifier.predict(x_test)

    accuracy = 100 * accuracy_score(y_test, predicted)                
    print('SVC accuracy: [%.2f]' % (accuracy ))

    #results = classifier.predict_proba(X_test)[0]

    #retorna formato de 28x28 das imagens usadas para teste
    images_restored = x_test.reshape((-1, image_size, image_size))
    images_and_predictions = list(zip(images_restored, predicted))

    plot_number(axes, images_and_predictions, line = 1, prelabel = 'Prediction')
    
    return classifier;


if __name__ == '__main__':
    digits_recognition()
    plt.show()
