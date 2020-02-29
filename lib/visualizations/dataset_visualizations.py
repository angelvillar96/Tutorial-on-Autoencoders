"""
Autoencoders/visualizations/dataset_visualizations.py

Auxiliary methods to visualize data from the datasets
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.colors import ListedColormap


def display_dataset_subset(image_set, num_images=9, grid_shape=(3,3), random=False):
    """"
    Displaying a grid with images from the given dataset

    Args:
    -----
    image_set: tuple
        Tuple containing all the images and possibly labes in the dataset
    num_images: integer
        number if images to display
    grid_shape: tuplex (X,Y)
        shape of the grid/matrix used to display the images
    random: boolean
        decides whether images are sampeld randomly or in a sorted fashion
    """

    if(random==True):
        idx = np.random.randint(low=0, high=len(image_set[1]), size=num_images)
    else:
        idx = np.arange(num_images)

    plt.figure(figsize=(grid_shape[1]*3,grid_shape[0]*3))
    for i in range(num_images):

        plt.subplot(*grid_shape, i+1)
        plt.imshow(image_set[0][idx[i],:])
        plt.title(f"Image n. {idx[i]}. Label {image_set[1][idx[i]][0]}")

    plt.tight_layout()
    plt.show()
