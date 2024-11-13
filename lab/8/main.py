import re
import pandas as pd
import string
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the LSTM model


class SentimentLSTM(nn.Module):
    def __init__(self, max_features, embed_dim, lstm_out, num_classes=2):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(max_features, embed_dim)
        self.lstm = nn.LSTM(embed_dim, lstm_out, batch_first=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(lstm_out, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = self.fc(h_n[-1])  # Using the last hidden state
        return out


# Define the GRU model
class SentimentGRU(nn.Module):
    def __init__(self, max_features, embed_dim, gru_out, num_classes=2):
        super(SentimentGRU, self).__init__()
        self.embedding = nn.Embedding(max_features, embed_dim)
        self.gru = nn.GRU(embed_dim, gru_out, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(gru_out, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        gru_out, h_n = self.gru(x)
        out = self.fc(h_n[-1])  # Using the last hidden state
        return out


# Define the Simple RNN model
class SentimentRNN(nn.Module):
    def __init__(self, max_features, embed_dim, rnn_out, num_classes=2):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(max_features, embed_dim)
        self.rnn = nn.RNN(embed_dim, rnn_out, batch_first=True)
        self.fc = nn.Linear(rnn_out, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        rnn_out, h_n = self.rnn(x)
        out = self.fc(h_n[-1])  # Using the last hidden state
        return out


# Load the Melbourne Temperature data
# Adjust the path
data = pd.read_csv('archive/daily-minimum-temperatures-in-me.csv', header=0)
temperatures = data['temp'].values.astype(float)

# Normalize the temperature values
# scaler = MinMaxScaler(feature_range=(0, 1))
# temperatures = scaler.fit_transform(temperatures.reshape(-1, 1)).squeeze()

# Set sequence length (number of previous time steps to use for prediction)
sequence_length = 1000

# Create sequences and labels
sequences = []
labels = []
for i in range(len(temperatures) - sequence_length):
    sequences.append(temperatures[i:i + sequence_length])
    labels.append(temperatures[i + sequence_length])

sequences = np.array(sequences)
labels = np.array(labels)

# Convert to tensors
sequences = torch.tensor(sequences, dtype=torch.float32).to(device)
labels = torch.tensor(labels, dtype=torch.float32).to(device)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    sequences, labels, test_size=0.2, random_state=42)

# Create DataLoader objects
train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)


# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for sequences, labels in train_loader:
            optimizer.zero_grad()
            output = model(sequences.long()).reshape((-1,))
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for sequences, labels in val_loader:
                output = model(sequences.long()).reshape((-1,))
                loss = criterion(output, labels)
                val_loss += loss.item()
        if epoch % 10 == 0:
            print(
                f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}')

# Define the evaluation function


def evaluate_model(model, data_loader):
    model.eval()
    predictions = []
    actuals = []

    # Generate predictions
    with torch.no_grad():
        for sequences, labels in data_loader:
            output = model(sequences.long()).reshape((-1,))  # Forward pass
            predictions.extend(output.cpu().numpy())  # Append predictions
            actuals.extend(labels.cpu().numpy())  # Append actual labels

    # Calculate MSE, RMSE, and MAE
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    print(f'MSE: {mse}, RMSE: {rmse}, MAE: {mae}')

    return actuals, predictions, mse, rmse, mae

# Define the plotting function


def plot_predictions(actuals, predictions, title='Model Predictions vs Actual'):
    plt.figure(figsize=(10, 5))
    plt.plot(actuals, label="Actual", color='b')
    plt.plot(predictions, label="Predicted", color='r')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Temperature (normalized)")
    plt.legend()
    plt.show()


# Model parameters
embed_dim = 128
hidden_size = 128
max_features = sequence_length
model = SentimentRNN(max_features, embed_dim, hidden_size,
                     num_classes=1).to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer)

# Evaluate on the validation set
actuals, predictions, mse, rmse, mae = evaluate_model(model, val_loader)
# Plot the first 100 predictions
plot_predictions(actuals[:100], predictions[:100])

# Model parameters
embed_dim = 64
hidden_size = 128
max_features = sequence_length
model = SentimentLSTM(max_features, embed_dim,
                      hidden_size, num_classes=1).to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer)

# Evaluate on the validation set
actuals, predictions, mse, rmse, mae = evaluate_model(model, val_loader)
# Plot the first 100 predictions
plot_predictions(actuals[:100], predictions[:100])


# Model parameters
embed_dim = 128
hidden_size = 128
max_features = sequence_length
model = SentimentGRU(max_features, embed_dim, hidden_size,
                     num_classes=1).to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer)

# Evaluate on the validation set
actuals, predictions, mse, rmse, mae = evaluate_model(model, val_loader)
# Plot the first 100 predictions
plot_predictions(actuals[:100], predictions[:100])
