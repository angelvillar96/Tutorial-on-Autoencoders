# Autoencoders and Deep Learning with Keras

Autoencoders are neural networks that aim at copying their inputs to their outputs. They have been widely used for several tasks such as dimensionality reduction, denoising or data generation.

This repository illustrates the Tutorial Posts on Autoencoders found [in my personal website](http://www.angelvillarcorrales.com/templates/tutorials/autoencoders/IntroductionAutoencoders.php)


## Getting Started

To get the code, fork this repository or clone it using the following command:

>git clone https://github.com/angelvillar96/Tutorial-on-Autoencoders.git


### Prerequisites

This repository requires Python version (at least) 3.6 and the following packages: Numpy, Sklearn, Matplotlib and Keras.

I recommend creating an Anaconda environment and installing the packages. This can be done by running the following commands in the Shell or Anaconda terminal.

```shell
$ conda create --name autoencoders python=3.6
$ activate autoencoders
$ conda install -c conda-forge numpy matplotlib keras jupyterlab
$ conda install scikit-learn
```

*__Note__:* This step might take a few minutes


## Contents

This repository contains 4 different Jupyter notebooks illustrating different types of autoencoders and some applications. These notebooks are self-contained, which means that they run independently of the other ones.

Part of the code is explicitely written in the notebook cells, but (for the sake of cleanliness) part of the code is called from the library files.


These library files are located under the /lib directory, and they include several useful methods for many different tasks such as instanciating Autoencoder models, loading and preprocessing data or visualizations. The classes and methods, including their arguments and returned values, are further explained in the files,
