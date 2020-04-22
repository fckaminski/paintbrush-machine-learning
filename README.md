# paintbrush-machine-learning
Paintbrush-like canvas to hand draw digits to then be fed into machine learning trained models. Includes SVC (support vector classifier) and CNN (Convolutional Neural Networks) leaning models.

Kaminski - fckaminski66@gmail.com

# Instructions:
1) digits_recognition_SVC.py: Trains MNIST dataset using sklearn SVC (support vector classifier). 
                              Saves model to file: "digits_recognition.pickle"


2) digits_recognition_keras_cnn.py: Trains MNIST dataset using CNN (Convolutional Neural Networks).
                                    Saves model to file: "digits_recognition_cnn.h5".

3) digits_recognition_canvas.py: Loads the two above models files and runs the canvas to paint the digits.
