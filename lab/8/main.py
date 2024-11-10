import re
import pandas as pd
import string
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess the data
data = pd.read_csv('IMDB-sentiment-analysis-master/IMDB-Dataset.csv')

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

# Build vocabulary manually and encode
max_features = 5000
counter = Counter([word for review in data['review']
                  for word in review.split()])
# Reserve spots for <pad> and <unk>
most_common = counter.most_common(max_features - 2)
# +2 to reserve 0 for <pad>, 1 for <unk>
vocab = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
vocab['<pad>'] = 0
vocab['<unk>'] = 1


def encode(text):
    return [vocab.get(token, vocab['<unk>']) for token in text.split()]


data['encoded_review'] = data['review'].apply(encode)

# Pad sequences to maxlen
maxlen = 600


def pad_sequence_custom(sequence, maxlen=maxlen, padding_value=vocab['<pad>']):
    return torch.tensor(sequence[:maxlen] + [padding_value] * (maxlen - len(sequence)), dtype=torch.long)


data['padded_review'] = data['encoded_review'].apply(pad_sequence_custom)
X = torch.stack(data['padded_review'].tolist())

# Convert labels to binary
data['sentiment'] = data['sentiment'].apply(
    lambda x: 1 if x == 'positive' else 0)
Y = torch.tensor(data['sentiment'].values, dtype=torch.long)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

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
model = SentimentLSTM(max_features, embed_dim, lstm_out).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
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
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Validation Accuracy: {100 * correct / total}%")
