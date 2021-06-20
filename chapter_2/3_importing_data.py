# Graham Williams
# grw400@gmail.com

# import data from nnfs datasets

import numpy as np
import nnfs

from nnfs.datasets import spiral_data

nnfs.init() #sets random seed to 0,creates float32 dtype default, overrides original dot product from numpy

import matplotlib.pyplot as plt

X, y = spiral_data(samples=100, classes=3)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()


