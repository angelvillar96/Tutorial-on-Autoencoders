"""
Autoencoders/models/visualizations.py

This file contains different autoencoder architectures based solely on fully-connected layers
"""

import numpy as np

import keras.backend as K
from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, Flatten, Reshape, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam


def basic_conv_autoencoder(bottleneck_dim=2, loss="binary_crossentropy"):
    """
    Creates a fixed convolutional neural network with just one conv layer and one hidden
    fully connected layer in encoder and decoder

    Args:
    -----
    bottleneck_dim: integer
        dimensionality of the latent space representation
    loss: string
        name of the loss function to use (mse, binary_crossentropy, ...)

    Returns:
    --------
    autoencoder: Model
        model corresponding to the autoencoder
    autoencoder: Model
        encoder part of the autoencoder
    """

    input_img = Input(shape=(28, 28, 1))

    # encoder
    features = Conv2D(32, (3, 3), padding="same", activation="relu")(input_img)
    pooled_features = MaxPooling2D((2,2), padding="same")(features)
    flattened_features = Flatten()(pooled_features)
    hidden_e = Dense(100, activation='relu')(flattened_features)
    latent_representation = Dense(bottleneck_dim, activation='relu', name="bottleneck")(hidden_e)

    # intermediate feature shapes
    flattened_shape = 14*14*32
    feature_shape = (14,14,32)

    # decoder
    hidden_d = Dense(100, activation='relu')(latent_representation)
    decoded = Dense(flattened_shape, activation='relu')(hidden_d)
    decoded = Reshape(target_shape=feature_shape)(decoded)
    decoded = UpSampling2D((2,2))(decoded)
    decoded = Conv2D(1, (3, 3), padding="same", activation="relu")(decoded)

    optimizer = Adam()
    autoencoder = Model(input_img, decoded, name="conv_autoencoder")
    encoder = Model(input_img, latent_representation, name="conv_encoder")
    autoencoder.compile(optimizer=optimizer, loss=loss)

    return autoencoder, encoder


#
