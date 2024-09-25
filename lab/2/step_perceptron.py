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
        return self.activation_function(linear_output)

    def activation_function(self, x):
        return np.where(x > 0, 1, 0)

    def compute_loss(self, y_true, y_pred):
        # Squared loss: (1/2) * (y_pred - y_true)^2
        loss = 0.5 * np.mean((y_pred - y_true.reshape(-1, 1)) ** 2)
        return loss

    def accuracy(self, x, y_true):
        y_pred = self.predict(x).flatten()
        return np.mean(y_pred == y_true)

    def backward(self, x, y_true, y_pred):
        # Compute the error
        error = y_true.reshape(-1, 1) - y_pred

        # Update weights and threshold (bias)
        self.w += self.lr * np.dot(x.T, error)  # Gradient w.r.t weights
        # Gradient w.r.t threshold (bias)
        self.threshold += self.lr * np.sum(error)

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
        return self.activation_function(y_pred).astype(int)


def plot_loss_and_accuracy(perceptron):
    epochs = range(len(perceptron.losses))

    # Plot Loss
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, perceptron.losses, label="Training Loss")
    plt.title("Loss vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, perceptron.train_accuracies,
             label="Training Accuracy", color='blue')
    plt.plot(epochs, perceptron.test_accuracies,
             label="Test Accuracy", color='green')
    plt.title("Accuracy vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Display plots
    plt.tight_layout()
    plt.show()
