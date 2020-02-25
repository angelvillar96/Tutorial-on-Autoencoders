"""
Autoencoders/utils/projections.py

This file contains different methods to compute projections onto low-dimensional
spaces using methods such as PCA or t-SNE.
"""

import numpy as np
from sklearn.decomposition import PCA


def computed_PCA_projection(train_data, test_data, dim=2):
    """
    Computing a projection onto a low-dimensional manifold using Principal Component Analysis

    Args:
    -----
    train_data: numpy array
        np array containign the data used to fit the PCA model
    test_data: numpy array
        np array with the  data that we wan to project onto the latent space
    dim: integer
        dimensionality of the latent space we project onto

    Return:
    -------
    low_dim_projections: numpy array
        low-dimensional projection of the test data
    """

    train_data = train_data.reshape(train_data.shape[0], np.prod(train_data.shape[1:]))
    test_data = test_data.reshape(test_data.shape[0], np.prod(test_data.shape[1:]))

    pca = PCA(n_components=dim)
    pca.fit(train_data)
    low_dim_projections = pca.transform(test_data)

    return low_dim_projections


#
