"""
Autoencoders/models/visualizations.py

This file contains different autoencoder architectures
"""

import numpy as np

import keras.backend as K
from keras.datasets import mnist
from keras.layers import Input, Dense, GaussianNoise, Flatten, Reshape, Lambda
from keras.models import Model
from keras.optimizers import Adam





##########################################
# fully connected network
def fully_connected(bottleneck_dim=2, stddev_noise=0.1, lr=0.1, loss="binary_crossentropy"):

    input_img = Input(shape=(28, 28,))

    encoded = Flatten()(input_img)
    encoded = Dense(256, activation='relu')(encoded)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(bottleneck_dim)(encoded)

    channel_input = Lambda(energy_normalization)(encoded)
    channel_output = GaussianNoise(stddev_noise)(channel_input)

    decoded = Dense(64, activation='relu')(channel_output)
    decoded = Dense(256, activation='relu')(decoded)
    decoded = Dense(784, activation='sigmoid')(decoded)
    decoded = Reshape(target_shape=(28, 28))(decoded)

    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, channel_input)

    opt = Adam(lr=lr)
    autoencoder.compile(optimizer=opt, loss=loss)

    return autoencoder, encoder


SNR_dB = 20
stddev_noise = compute_noise_power(SNR_dB)
print(f"Std Dev Noise: {stddev_noise}")

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
training_size = x_train.shape
test_size = x_test.shape
print(f"Training set size: {training_size}\nTest set size: {test_size}")


autoencoder, encoder = fully_connected(bottleneck_dim=3, stddev_noise=stddev_noise,
                              lr=0.01, loss="binary_crossentropy")

autoencoder.fit(x_train, x_train,
                epochs=20,
                batch_size=1024,
                shuffle=True,
                validation_data=(x_test, x_test))


# encoding and decoding some digits
decoded_imgs = autoencoder.predict(x_test)
random_numbers = np.random.randint(0, test_size[-1], 5)
fig, ax = plt.subplots(2,5)
fig.set_size_inches(16,6)
for i,n in enumerate(random_numbers):
    ax[0,i].imshow(x_test[n])
    ax[1,i].imshow(decoded_imgs[n])


# displayinng the latent space
coded_imgs = encoder.predict(x_test)
plot_latent(coded_imgs, y_test)
