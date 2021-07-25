# Softmax activation
class Activation_Softmax:
    ...
    # Backward pass
    def backward(self, dvalues):

        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
            enumerate(zip(self.output, dvalues)):

            # Flatten output array
            single_output = single_output.reshape(-1,1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues)

# First, create empty array (becomes gradient) w/ same shape as grads
# receiving to apply the chain rule.

# Next, iterate sample-wise over pairs of the outputs and gradients,
# calculating partial derivatives and calc'ing final product of Jacobian/gradient vector
# storing resulting row vector in the dinput array