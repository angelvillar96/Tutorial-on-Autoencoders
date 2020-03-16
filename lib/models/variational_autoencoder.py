"""
Autoencoders/models/variational_autoencoders.py

This file contains the implementation of variational autoencoders and necessary methods
for training, inference and generation such as sampling or loss functions
"""

import numpy as np

import keras.backend as K
from keras.datasets import mnist
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers.core import Lambda
from keras.models import Model
from keras.optimizers import Adam


def variational_autoencoder(bottleneck_dim=30, input_shape=(28,28,)):
    """
    Creates an autoencoder with just one hidden layer in encoder and decoder

    Args:
    -----
    bottleneck_dim: integer
        dimensionality of the multivariate gaussian. Corresponds to the number of
        elements in the mean vector and in the diagonal of the covariance matrix
    input_shape: tuple (X,Y)
        tuple with the shape of the input images

    Returns:
    --------
    autoencoder: Model
        model corresponding to the variational autoencoder
    encoder: Model
        encoder part of the VAE
    generator: Model
        decoder part of the VAE, which can be used for generative purposes
    """

    input_img = Input(shape=input_shape)

    flattened_shape = np.prod(input_img.shape[1:])
    target_shape = input_img.shape[1:]

    # encoder
    flattened_input = Flatten()(input_img)
    hidden_e = Dense(256, activation='relu')(flattened_input)
    hidden_e = Dense(128, activation='relu')(hidden_e)
    mean_representation = Dense(bottleneck_dim, activation='relu', name="mean")(hidden_e)
    variance_representation = Dense(bottleneck_dim, activation='relu', name="variance")(hidden_e)

    # taking a sample
    sample = Lambda(taking_sample, output_shape=(bottleneck_dim,))([mean_representation, variance_representation])

    # decoder
    hidden_d = Dense(128, activation='relu')(sample)
    hidden_d = Dense(256, activation='relu')(hidden_d)
    hidden_d = Dense(flattened_shape, activation='relu')(hidden_d)
    decoded = Reshape(target_shape=target_shape)(hidden_d)

    optimizer = Adam()
    autoencoder = Model(input_img, decoded, name="variational_autoencoder")
    encoder = Model(input_img, sample, name="encoder")
    generator = Model(input_img, decoded, name="generator")

    # compiling using our custom loss
    autoencoder.compile(optimizer=optimizer, loss=vae_loss_function)

    return autoencoder, encoder, generator


def vae_loss_function(input, output):
    """
    Special variational autoencoder loss, which corresponds to a regularized MSE

    Args:
    -----
    input: tensor
        inputs fed to the neural network
    output: tensor
        tensors reconstructd by the autoencoder

    Returns:
    --------
    loss: float
        loss corresponding to the batch
    """

    # reconstruction error corresponding to the expectation
    print(input.shape)
    print(output.shape)
    reconstruction_error = K.sum( K.mean(K.square(input-output)), axis=1)

    # regularization error corresponding to the KL divergence
    error = K.exp(variance_representation) + K.square(mean_representation) - 1. - variance_representation
    regularization_error = K.sum(error, axis=1)

    loss = reconstruction_error + regularization_error

    return loss


def taking_sample(args):
    """
    Taking a sample from the Gaussian distribution spanned by the mean and covariance
    output by the encoder

    Args:
    -----
    args: list
        list containing two elements:
            1.- mean of the Z distribution output by the encoder
            2.- diagonal of the covariance matrix of the Z distribution

    Returns:
    --------
    sample: tensor
        sample from the gaussian N(0,1)
    """

    mean, variance = args
    batch_size, latent_dim = mean.shape
    if(batch_size==None):
        batch_size = 0

    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
    sample = mean + K.exp(variance / 2) * epsilon

    return sample

#
