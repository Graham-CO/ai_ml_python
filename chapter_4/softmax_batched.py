# Graham Williams
# grw400@gmail.com

# Tune softmax activation function class to accept batch inputs

import numpy as np

class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):

        # Get unnormalized probabilities & subtract largest value to avoid dead/exploding neurons
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # axis = 1 sums along rows (0 is along columns for 2D arr), keepdims=True retains it as column vector

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.ouput = probabilities