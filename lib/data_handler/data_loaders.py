"""
Autoencoders/data_handler/data_loaders.py

This file contains different methods to load, handle and preprocess the data
"""

from keras.datasets import mnist

def load_mnist_dataset():
    """
    Loading train and test set of the MNIST dataset
    """

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    return (x_train, y_train), (x_test, y_test)
