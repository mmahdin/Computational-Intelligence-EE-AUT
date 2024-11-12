import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transfer_function(z):
    numerator = -z**2 + 1.9 * z + 0.95
    denominator = z**3 - 0.18 * z**2 + 0.08 * z - 0.08
    return numerator / denominator


def generate_delayed_data(data, delays):
    delayed_data = []
    for delay in delays:
        padded_data = torch.cat(
            [torch.zeros(delay, device=device), data[:-delay]])
        delayed_data.append(padded_data)
    return torch.stack(delayed_data, dim=1)


class StaticEstimator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(StaticEstimator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        return self.fc2(x)


class SystemIdentifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SystemIdentifier, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # Output layer

        # Activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        # Forward pass through the network
        x = self.relu(self.fc1(x))  # First layer with ReLU activation
        x = self.relu(self.fc2(x))  # Second layer with ReLU activation
        # Output layer (no activation for regression)
        x = self.fc3(x)
        return x


class DeepNetwork(torch.nn.Module):
    def __init__(self, input_dim=8, hidden_dims=[128, 128, 128], num_classes=5):
        super(DeepNetwork, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims

        # Build the hidden layers
        for i in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            layers.append(torch.nn.ReLU())

        # Output layer
        layers.append(torch.nn.Linear(dims[-1], num_classes))

        # Sequentially stack the layers
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Example white noise input signal
input_signal = torch.randn(100, device=device)
input_delays = [1, 2]
output_delays = [1, 2]

# Generate delayed inputs and outputs
input_data_d = generate_delayed_data(input_signal, input_delays)
input_data = torch.cat([input_signal.reshape(-1, 1), input_data_d], dim=1)
output_data = generate_delayed_data(
    transfer_function(input_signal), output_delays)


# Initialize the model, loss, and optimizer
# model = SystemIdentifier(input_data.shape[1] + output_data.shape[1], 512, 1).to(device)
model = DeepNetwork(input_dim=input_data.shape[1] + output_data.shape[1], hidden_dims=[
                    128, 64, 64], num_classes=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Prepare the data
X = torch.cat([input_data, output_data], dim=1).to(device)
y = transfer_function(input_signal).view(-1, 1).to(device)

# Training loop
num_epochs = 50000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if epoch % 5000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Test the model with a new input signal (e.g., a sine wave)
test_input = torch.sin(torch.linspace(0, 10, 100, device=device))
test_input_data = generate_delayed_data(test_input, input_delays)
test_input_data = torch.cat(
    [test_input.reshape(-1, 1), test_input_data], dim=1)
test_output_data = generate_delayed_data(
    transfer_function(test_input), output_delays)
X_test = torch.cat([test_input_data, test_output_data], dim=1).to(device)

# Combine delayed test input and output for prediction
X_test = torch.cat([test_input_data, test_output_data], dim=1).to(device)
predicted_output = model(X_test)

# Plot predicted output and actual output
plt.plot(predicted_output.detach().cpu().numpy(),
         label="Predicted Output", color="blue")
plt.plot(transfer_function(test_input).cpu().numpy(),
         label="Actual Output", color="red", linestyle="--")
plt.xlabel("Time")
plt.ylabel("Output")
plt.legend()
plt.title("Predicted vs. Actual Output")
plt.show()
