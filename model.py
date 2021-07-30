#!/usr/bin/env python3

"""Model to classify draft beers

This file contains all the model information: the training steps, the batch
size and the model itself.
"""

import tensorflow as tf
from tensorflow.keras import models, layers

def get_batch_size():
    """Returns the batch size that will be used by your solution.
    It is recommended to change this value.
    """
    return 32

def get_epochs():
    """Returns number of epochs that will be used by your solution.
    It is recommended to change this value.
    """
    return 100

def solution(input_layer):
    """Returns a compiled model.

    This function is expected to return a model to identity the different beers.
    The model's outputs are expected to be probabilities for the classes and
    and it should be ready for training.
    The input layer specifies the shape of the images. The preprocessing
    applied to the images is specified in data.py.

    Add your solution below.

    Parameters:
        input_layer: A tf.keras.layers.InputLayer() specifying the shape of the input.
            RGB colored images, shape: (width, height, 3)
    Returns:
        model: A compiled model
    """
    
    input_shape = (160, 160, 3)

    # TODO: Code of your solution
    model = models.Sequential()
    model.add(layers.Conv2D(160, (3,3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))

    model.add(layers.Flatten())
    # TODO: make fully connected layer
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(4, activation = 'softmax'))

    print(model.summary())


    # TODO: Return the compiled model
    # TODO: read on which optimizer and loss function to use
    model.compile(tf.optimizers.RMSprop(0.001), loss='mse', metrics=['accuracy'])
    return model
