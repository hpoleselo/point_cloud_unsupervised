"""
Taken and adapted from:
https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/
"""

from numpy import asarray
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D

# Binary image with a single channel and a two-pixel wide vertical
# line on the center
data = [[0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0]]

data = asarray(data)

# Reshaping to a single channel image
data = data.reshape(1, 8, 8, 1)

# Creates model
model = Sequential()

# Adds one layer containing the ReLU function to be applied to each value
# in the feature map.
model.add(Conv2D(1, (3,3), activation='relu', input_shape=(8, 8, 1)))