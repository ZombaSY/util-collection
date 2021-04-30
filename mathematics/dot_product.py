import numpy as np

array_a = np.array([0, 1, 0, 0, 0], dtype=np.float)     # one-hot vector
array_b = np.array([0, 10, 0, 0, 0], dtype=np.float)   # sum should be 1

array_a = array_a / array_a.sum(axis=0)
array_b = array_b / array_b.sum(axis=0)

result = np.dot(array_a, array_b.transpose())   # dot product

print(result)
