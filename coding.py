from tkinter import HORIZONTAL
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib
from tensorflow.keras.datasets import cifar10
from symtable import factor

num_classes = 10
input_shape = (32,32,3)  #32x32 size of images; 3 layers of RGB Color

#import dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
print(f"x_train shape: {x_train.shape} - y_train.shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

#defining learning parameters
learning_rate = 0.001        #it specifies the step size at which the model weights are updated, smaller the step size; the samller the learning rate
weight_decay = 0.0001        #it is the regularization technique that adds a penalty term to the loss functions to prevent over-fitting, it basically determines the strength of the penalty
batch_size = 256             #it refers to the training examples we are going to use with each iteration
num_epochs = 40              #it is the number of times we are going to train the entire data
image_size = 72              #resizing the input images
patch_size = 6               #it determines the size of the patches that we are going to extract from the input images
num_heads = 4                #to determine the number of attention head (it allows model to focus in the different parts of the inuts simultaneously)
projection_dim = 64
transformer_units = [projection_dim*2, projection_dim]      #it represents the size of the transformer layers in terms of dimensions
transformer_layers = 8
mlp_head_unit = [2048, 1024]     #size of the dense layers of the final classifie

#DATA AUGMENTATION - it is a technique to artificially increase the size of the training dataset to help us improve the model generalization
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        #mirroring the image w.r.t vertical axes, so we need to do the horizontal flip
        layers.RandomFlip('HORIZONTAL'),
        layers.RandomRotation(factor = 0.02),
        layers.RandomZoom(height_factor = 0.2, width_factor = 0.2)
    ],
    name = "data_augmentation"
)
#calculating MEAN and VARIANCE of our data for normalization
data_augmentation.layers[0].adapt(x_train)