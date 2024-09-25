import numpy as np


def sigmoid(x):
    """Sigmoid function"""
    return 2 / (1 + np.exp(-x)) - 1


def tanh(x):
    """tanh function"""
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))


def sigmoid_derivative(x):
    """Derivative of sigmoid function"""
    sig = sigmoid(x)
    return (1 - sig**2)/2


def sigmatrix(matrix, r=None, c=None):
    """Function to process a matrix"""
    result1 = 0
    result2 = 0
    # Loop over each element in the matrix
    for row in matrix:
        for element in row:
            result1 += sigmoid(element)
            result2 += sigmoid_derivative(element)
    return (result1, result2)


def tanhmatrix(matrix):
    """Function to process a matrix"""
    result = 0
    for row in matrix:
        for element in row:
            result += tanh(element)
    return result


# Test the function with the given matrix
matrix = np.array([[1, 0, np.sin(np.pi / 4)],
                  [0, 1, np.sin(np.pi / 2)], [1, 0, 1]])
output1 = sigmatrix(matrix)
output2 = tanhmatrix(matrix)

print(output1)
print(output2)
