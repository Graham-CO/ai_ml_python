## Categorical Cross Entropy, Dynamically Calculated
# Graham Williams
# grw400@gmail.com

import numpy as np

# 3 samples, 3 classes
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

# class 0 = dog | class 1 = cat | class 2 = human
# target = [dog, cat, cat]
class_targets = [0, 1, 1]

# for the 0th row, return the 0th idx 
# for the 1st row, return the 1st idx
# for the 2nd row, return the 1st idx
print(softmax_outputs[[0, 1, 2], class_targets])

# len(softmax_outputs) = 3
# range(3) = 0,1,2
print(softmax_outputs[
    range(len(softmax_outputs)), class_targets
])

# print a list of the confidences at the target indices for each sample
print(-np.log(softmax_outputs[
    range(len(softmax_outputs)), class_targets
]))

# map indices to retrieve values from softmax distributions
# zip() lets us iterate over multiple iterables at the same time
for targ_idx, distribution in zip(class_targets, softmax_outputs):
    print(distribution[targ_idx])

# apply negative log to confidences at target indices
neg_log = -np.log(softmax_outputs[
    range(len(softmax_outputs)), class_targets
])

# avg loss per batch, using arithmetic mean
average_loss = np.mean(neg_log)
print(average_loss)