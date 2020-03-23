"""
Autoencoders/models/variational_autoencoders.py

This file contains the implementation of variational autoencoders and necessary methods
for training, inference and generation such as sampling or loss functions
"""

import numpy as np

import keras.backend as K
from keras.datasets import mnist
from keras.losses import mse
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
    mean_representation = Dense(bottleneck_dim, name="mean")(hidden_e)
    variance_representation = Dense(bottleneck_dim, name="variance")(hidden_e)

    # taking a sample
    sample = Lambda(taking_sample, output_shape=(bottleneck_dim,), name="taking_sample")([mean_representation, variance_representation])

    # decoder
    hidden_d = Dense(128, activation='relu')(sample)
    hidden_d = Dense(256, activation='relu')(hidden_d)
    hidden_d = Dense(flattened_shape, activation='sigmoid')(hidden_d)
    decoded = Reshape(target_shape=target_shape)(hidden_d)

    # defining the models
    autoencoder = Model(input_img, decoded, name="variational_autoencoder")
    encoder = Model(input_img, [mean_representation, variance_representation, sample], name="encoder")
    generator = Model(input_img, decoded, name="generator")

    # compiling using our custom loss
    optimizer = Adam()
    vae_loss = vae_loss_function(input_img, decoded, mean_representation, variance_representation)

    autoencoder.add_loss(vae_loss)
    autoencoder.compile(optimizer=optimizer,)

    return autoencoder, encoder, generator



def vae_loss_function(input, output, mean, variance):
    """
    Custom regularized loss function used for the variational autonecoder

    Args:
    -----
    input: tensor
        input to the vae model
    output: tensor
        output of the vae model
    mean: tensor
        mean of the Z distribution output by the encoder
    variance: tensor
        diagonal of the covariance matrix of the Z distribution

    Returns:
    --------
    vae_loss: float
        total loss of the given batch
    """

    # loss contribution from the reconstruction error
    input_dim = K.shape(K.batch_flatten(input))[1]
    reconstruction_error = mse(K.batch_flatten(input), K.batch_flatten(output))*K.cast(input_dim,"float32")

    # regularizer to enforce our desired distribution
    regularization_error = 0.5*(K.exp(variance) + K.square(mean) - 1. - variance)
    regularization_error = K.sum(regularization_error, axis=-1)

    # total loss
    vae_loss = K.mean(reconstruction_error + regularization_error)

    return vae_loss


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
    batch_size = K.shape(mean)[0]
    latent_dim = K.shape(mean)[1]

    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
    sample = mean + K.exp(variance / 2) * epsilon

    return sample

#
