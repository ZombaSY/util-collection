import numpy as np

"""
http://wangshusen.github.io/code/countsketch.html
"""

matrixA = np.random.random([16, 100])
print('The matrix A has ', matrixA.shape[0], ' rows and ', matrixA.shape[1], ' columns.')


def countSketchInMemroy(matrixA, s):
    m, n = matrixA.shape
    matrixC = np.zeros([m, s])
    hashedIndices = np.random.choice(s, n, replace=True)

    randSigns = np.random.choice(2, n, replace=True) * 2 - 1 # a n-by-1{+1, -1} vector
    matrixA = matrixA * randSigns.reshape(1, n) # flip the signs of 50% columns of A
    for i in range(s):
        idx = (hashedIndices == i)
        matrixC[:, i] = np.sum(matrixA[:, idx], 1)

    return matrixC


s = 10 # sketch size, can be tuned
matrixC = countSketchInMemroy(matrixA, s)

print(matrixA)
print(matrixC)

# Test
# compare the l2 norm of each row of A and C
# rowNormsA = np.sqrt(np.sum(np.square(matrixA), 1))
# print(rowNormsA)
# rowNormsC = np.sqrt(np.sum(np.square(matrixC), 1))
# print(rowNormsC)


def countSketchStreaming(matrixA, s):
    m, n = matrixA.shape
    matrixC = np.zeros([m, s])
    hashedIndices = np.random.choice(s, n, replace=True)
    randSigns = np.random.choice(2, n, replace=True) * 2 - 1
    for j in range(n):
        a = matrixA[:, j]
        h = hashedIndices[j]
        g = randSigns[j]
        matrixC[:, h] += g * a
    return matrixC


s = 50 # sketch size, can be tuned
matrixC = countSketchStreaming(matrixA, s)
