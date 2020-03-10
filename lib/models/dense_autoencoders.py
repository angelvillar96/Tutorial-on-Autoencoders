"""
Autoencoders/models/dense_autoencoders.py

This file contains different autoencoder architectures based solely on fully-connected layers
"""

import numpy as np

import keras.backend as K
from keras.datasets import mnist
from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model
from keras.optimizers import Adam



def shallow_fully_connected_autoencoder(bottleneck_dim=2, layer_size=100, loss="binary_crossentropy"):
    """
    Creates an autoencoder with just one hidden layer in encoder and decoder

    Args:
    -----
    bottleneck_dim: integer
        dimensionality of the latent space representation
    layer_size: integer
        dimensionality of the hidden layer
    loss: string
        reconstruction loss function to use (binary_crossentropy, mse, ...)

    Returns:
    --------
    autoencoder: Model
        model corresponding to the autoencoder
    encoder: Model
        encoder part of the autoencoder
    """

    input_img = Input(shape=(28, 28,))
    flattened_shape = np.prod(input_img.shape[1:])
    target_shape = input_img.shape[1:]

    # encoder
    flattened_input = Flatten()(input_img)
    hidden_e = Dense(layer_size, activation='relu')(flattened_input)
    latent_representation = Dense(bottleneck_dim, activation='relu', name="bottleneck")(hidden_e)

    # decoder
    hidden_d = Dense(layer_size, activation='relu')(latent_representation)
    decoded = Dense(flattened_shape, activation='sigmoid')(hidden_d)
    decoded = Reshape(target_shape=target_shape)(decoded)

    optimizer = Adam()
    autoencoder = Model(input_img, decoded, name="shallow_autoencoder")
    encoder = Model(input_img, latent_representation, name="shallow_encoder")
    autoencoder.compile(optimizer=optimizer, loss=loss)

    return autoencoder, encoder


#
