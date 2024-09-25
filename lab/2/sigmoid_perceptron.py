import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, input_size=30, lr=0.01) -> None:
        self.input_size = input_size
        self.threshold = np.random.rand()
        self.w = np.random.rand(self.input_size, 1)
        self.lr = lr
        self.losses = []  # List to store loss at each epoch
        self.train_accuracies = []  # Track training accuracy
        self.test_accuracies = []  # Track test accuracy

    def forward(self, x):
        # Linear combination
        linear_output = np.dot(x, self.w) + self.threshold
        return self.sigmoid(linear_output)

    def sigmoid(self, out):
        return 1 / (1 + np.exp(-out))

    def sigmoid_derivative(self, out):
        # Derivative of the sigmoid function
        return out * (1 - out)

    def compute_loss(self, y_true, y_pred):
        # Squared loss: (1/2) * (y_pred - y_true)^2
        loss = 0.5 * np.mean((y_pred - y_true.reshape(-1, 1)) ** 2)
        return loss

    def accuracy(self, x, y_true):
        y_pred = self.predict(x).flatten()
        return np.mean(y_pred == y_true)

    def backward(self, x, y_true, y_pred):
        # Compute the error
        error = y_pred - y_true.reshape(-1, 1)

        # Derivative of sigmoid (for the output layer)
        d_pred = error * self.sigmoid_derivative(y_pred)

        # Update weights and threshold (bias)
        self.w -= self.lr * np.dot(x.T, d_pred)  # Gradient w.r.t weights
        # Gradient w.r.t threshold (bias)
        self.threshold -= self.lr * np.sum(d_pred)

    def train(self, x_train, y_train, x_test, y_test, epochs=100):
        for epoch in range(epochs):
            y_pred_train = self.forward(x_train)

            # Compute training loss and store it
            train_loss = self.compute_loss(y_train, y_pred_train)
            self.losses.append(train_loss)

            # Calculate accuracy for training and test data
            train_accuracy = self.accuracy(x_train, y_train)
            test_accuracy = self.accuracy(x_test, y_test)

            # Store accuracies
            self.train_accuracies.append(train_accuracy)
            self.test_accuracies.append(test_accuracy)

            # Update weights
            self.backward(x_train, y_train, y_pred_train)

    def predict(self, x):
        y_pred = self.forward(x)
        # Threshold at 0.5 to get binary output
        return (y_pred > 0.5).astype(int)
