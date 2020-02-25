"""
Autoencoders/visualizations/visualizations.py

This file contains different auxiliary methods to compute visualizations
of images, latent spaces and so on
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.colors import ListedColormap



def display_original_and_decoded_imgs(img_original, img_decoded, n_images, idx=None, title=""):
    """
    Displaying original input images and the same images after being encoded and decoded
    """

    if(idx==None):
        indices = np.random.randint(0, img_original.shape[0], n_images)
    else:
        indices = idx
    fig, ax = plt.subplots(2,n_images)
    fig.set_size_inches(16,6)
    plt.suptitle(title)
    for i,n in enumerate(indices):
        ax[0,i].imshow(img_original[n,:])
        ax[1,i].imshow(img_decoded[n,:])

    return


def plot_latent(points, label, dim=3, figsize=(10,6), title=""):
    """
    Displaying a 2- or 3-dimensional latent space and plotting the dat points
    """

    cb_labels = [str(i) for i in range(10)]
    colors = ['k', 'brown', 'r', 'orange', 'y', 'g', 'b', 'purple', 'grey', 'c']

    fig = plt.figure(figsize=figsize)

    if(dim == 3):
        ax = plt.axes(projection="3d")
        im = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=label, cmap=ListedColormap(colors))
    elif(dim == 2):
        ax = plt.axes()
        im = ax.scatter(points[:, 0], points[:, 1], c=label, cmap=ListedColormap(colors))
    else:
        print(f"Error! Dimension value must be 2 or 3 and it was set as {dim}")
        exit()

    ax.set_title(title)
    cb = fig.colorbar(im)
    loc = np.arange(0, max(label)+1, max(label) / float(len(colors))) + 0.4

    cb.set_ticks(loc)
    cb.set_ticklabels(cb_labels)

    plt.show()
