# Graham Williams
# grw400@gmail.com

# Apply ReLU activation function to a dense layer

import numpy as np
import nnfs

from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)

        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    
    # Forward pass
    def forward(self, inputs):
        # Calculate output values from input
        self.output = np.maximum(0, inputs)

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense Layer with 2 input features and 3 output values
dense1 = Layer_Dense(2,3)

# Create ReLU activation
activation1 = Activation_ReLU()

# Forward pass training data through layer
dense1.forward(X)

# Forward pass through activation func.
# Takes in output from previous layer
activation1.forward(dense1.output)

print(activation1.output[:5])