"""
Taken and adapted from:
https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/
"""

from numpy import asarray
from keras.models import Sequential
from keras.layers import Conv2D

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

# Summarizes model
model.summary()

# Filter
# Best would be to set the weights randomly, but we're hardcoding it.
# It's a 3x3 filter so that when it detects a vertical line it will
# strongly activate.
detector = [[[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]]]
weights = [asarray(detector), asarray([0.0])]

# Store the weights in the model
model.set_weights(weights)

# Apply filter to input data
yhat = model.predict(data)

# enumerate rows
for r in range(yhat.shape[1]):
    # print each column in the row
    print([yhat[0,r,c,0] for c in range(yhat.shape[2])])