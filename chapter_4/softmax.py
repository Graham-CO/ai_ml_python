# Graham Williams
# grw400@gmail.com

# softmax activation function definition

import numpy as np
import math

# Hand definition
layer_outputs = [4.8, 1.21, 2.385]

E = math.e 

# For each value in a vector, calculate the exponential 
exp_values = []
for output in layer_outputs:
    exp_values.append(E ** output)
print('exponentiated values:')
print(exp_values)

# Normalize values
norm_base = sum(exp_values)
norm_values = []
for value in exp_values:
    norm_values.append(value / norm_base)
print('Normalized exponentiated values:')
print(norm_values)

print('Sum of normalized values:', sum(norm_values))

# Definition using numpy

# calc exp val
exp_values = np.exp(layer_outputs)
print('exponentiated values:')
print(exp_values)

# normalize vals
norm_values = exp_values / np.sum(exp_values)
print('normalized exponentiated values:')
print(norm_values)
print('sum of normalized values:', np.sum(norm_values))




