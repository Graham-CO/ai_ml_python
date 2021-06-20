# Graham Williams
# grw400@gmail.com

import numpy as np
import nnfs

from nnfs.datasets import spiral_data

nnfs.init()

# create a dense layer class and perform a forward pass through NN

class Layer_Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) # random number from normal distribution - mean 0, variance 0.01 (so range of -0.03 <--> 0.03, 99.7% CI - I think) || (inputs,neurons) to avoid transposing at every fwd pass
                                                                   # N(mu,sigma^2) --> sigma * np.random.randn(...) + mu
        self.biases = np.zeros((1, n_neurons)) # 0 vector

    # Forward pass
    def forward(self, inputs):
        # Calculate outputs from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense Layer w/ 2 input features and 3 output values
dense1 = Layer_Dense(2,3)

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# First few samples:
print(dense1.output[:5])

