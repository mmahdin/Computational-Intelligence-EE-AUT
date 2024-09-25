import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Perceptron:
    def __init__(self, input_size=30, lr=0.01) -> None:
        self.input_size = input_size
        self.threshold = np.random.rand()
        self.w = np.random.rand(self.input_size, 1)
        self.lr = lr

    def forward(self, x):
        # Linear combination
        linear_output = np.dot(x, self.w) + self.threshold
        return self.sigmoid(linear_output)

    def sigmoid(self, out):
        return 1 / (1 + np.exp(-out))

    def sigmoid_derivative(self, out):
        # Derivative of the sigmoid function
        return out * (1 - out)

    def backward(self, x, y_true, y_pred):
        # Compute the error
        error = y_pred - y_true.reshape(-1, 1)

        # Derivative of sigmoid (for the output layer)
        d_pred = error * self.sigmoid_derivative(y_pred)

        # Update weights and threshold (bias)
        self.w -= self.lr * np.dot(x.T, d_pred)  # Gradient w.r.t weights
        # Gradient w.r.t threshold (bias)
        self.threshold -= self.lr * np.sum(d_pred)

    def train(self, x, y_true, epochs=100):
        for epoch in range(epochs):
            y_pred = self.forward(x)
            self.backward(x, y_true, y_pred)

    def predict(self, x):
        y_pred = self.forward(x)
        # Threshold at 0.5 to get binary output
        return (y_pred > 0.5).astype(int)


# Load the dataset
cancer_data = load_breast_cancer()

# Features (input data)
X = cancer_data.data

# Labels (output data)
y = cancer_data.target

# Normalize the data for faster convergence
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Initialize the Perceptron
perceptron = Perceptron(input_size=X_train.shape[1], lr=0.01)

# Train the Perceptron
perceptron.train(X_train, y_train, epochs=1000)

# Predict on the test set
y_pred = perceptron.predict(X_test)

# Calculate accuracy
accuracy = np.mean(y_pred.flatten() == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
