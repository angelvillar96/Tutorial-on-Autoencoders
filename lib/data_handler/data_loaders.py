"""
Autoencoders/data_handler/data_loaders.py

This file contains different methods to load, handle and preprocess the data
"""

from keras.datasets import mnist, cifar10

def load_mnist_dataset():
    """
    Loading train and test set of the MNIST dataset
    """

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    return (x_train, y_train), (x_test, y_test)


def prepare_mnist_for_conv(x_train, x_test):
    """
    Reshapes the data from MNIST train and test sets to be fed into a convolutional
    neural network

    Args:
    -----
    x_train: np array
        original images from the MNIST train set
    x_test: np array
        original images from the MNIST test set
    Returns:
    --------
    x_train: np array
        reshaped images from the MNIST train set
    x_test: np array
        reshaped images from the MNIST test set
    """

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    return x_train, x_test


def load_ciphar_data():
    """
    Loading train and test sets of the CIFAR-10 dataset
    """

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    return (x_train, y_train), (x_test, y_test)
