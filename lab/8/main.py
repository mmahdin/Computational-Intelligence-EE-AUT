import re
import pandas as pd
import string
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Load and preprocess the data
data = pd.read_csv('IMDB-Dataset.csv')

# Removing HTML tags


def clean_html(text):
    clean = re.compile('<.*?>')
    cleantext = re.sub(clean, '', text)
    return cleantext

# First round of cleaning


def clean_text1(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Second round of cleaning


def clean_text2(text):
    text = re.sub('[''"",,,]', '', text)
    text = re.sub('\n', '', text)
    return text


data['review'] = data['review'].apply(
    clean_html).apply(clean_text1).apply(clean_text2)

# Tokenize and pad sequences
max_features = 5000
maxlen = 600
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['review'].values)
X = tokenizer.texts_to_sequences(data['review'].values)
X = pad_sequences(X, maxlen=maxlen)

# Convert labels to categorical
data['sentiment'] = data['sentiment'].apply(
    lambda x: 1 if x == 'positive' else 0)
Y = data['sentiment'].values

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.long)
Y_train = torch.tensor(Y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.long)
Y_test = torch.tensor(Y_test, dtype=torch.long)

# Create DataLoader for batching
batch_size = 64
train_data = TensorDataset(X_train, Y_train)
test_data = TensorDataset(X_test, Y_test)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

# Define the LSTM model


class SentimentLSTM(nn.Module):
    def __init__(self, max_features, embed_dim, lstm_out, num_classes=2):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(max_features, embed_dim)
        self.lstm = nn.LSTM(embed_dim, lstm_out, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(lstm_out, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = self.fc(h_n[-1])  # Using the last hidden state
        return out


# Model parameters
embed_dim = 128
lstm_out = 128
model = SentimentLSTM(max_features, embed_dim, lstm_out)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 16
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

# Validation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Validation Accuracy: {100 * correct / total}%")
