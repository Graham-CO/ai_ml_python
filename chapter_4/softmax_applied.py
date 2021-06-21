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

class Activation_Softmax:
    
    # Forward pass
    def forward(self, inputs):

        # Get unnormalized probabilities & subtract largest value to avoid dead/exploding neurons
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # axis = 1 sums along rows (0 is along columns for 2D arr), keepdims=True retains it as column vector

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Dense layer w/ 2 inp feature & 3 out vals
dense1 = Layer_Dense(2,3)

# Create ReLU activation for 1st dense layer
activation1 = Activation_ReLU()

# Create 2nd dense layer w/ 3 inp feature & 3 out vals
dense2 = Layer_Dense(3,3)

# Create softmax activation for 2nd dense layer
activation2 = Activation_Softmax()

# Forward pass through 1st layer
dense1.forward(X)

# Forward pass through 1st activation function
activation1.forward(dense1.output)

# Forward pass through 2nd dense layer
dense2.forward(activation1.output)

# Forward pass through 2nd act function
activation2.forward(dense2.output)

print(activation2.output[:5])