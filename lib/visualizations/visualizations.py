"""
Autoencoders/utils/visualizations.py

This file contains different auxiliary methods to compute visualizations
of images, latent spaces and so on
"""

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.colors import ListedColormap



def plot_latent(points, label, dim=3, figsize=(10,8)):
    """
    Displaying a 2- or 3-dimensional latent space and plotting the dat points
    """

    cb_labels = [str(i) for i in range(10)]
    colors = ['k', 'brown', 'r', 'orange', 'y', 'g', 'b', 'purple', 'grey', 'white']

    fig = plt.figure(figsize=figsize)

    if(dim == 3):
        ax = plt.axes(projection="3d")
        im = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=y_test, cmap=ListedColormap(colors))
    elif(dim == 2):
        ax = plt.axes()
        im = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=y_test, cmap=ListedColormap(colors))
    else:
        print(f"Error! Dimension value must be 2 or 3 and it was set as {dim}")
        exit()


    cb = fig.colorbar(im)
    loc = np.arange(0, max(y_test), max(y_test) / float(len(colors)))

    cb.set_ticks(loc)
    cb.set_ticklabels(cb_labels)

    plt.show()
