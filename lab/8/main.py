import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, random_split, DataLoader
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
from sklearn.metrics import accuracy_score, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Custom dataset class with padding
class SentimentAnalysisDataset(Dataset):
    def __init__(self, csv_file, max_length=2494):
        # Load data from CSV
        self.data = pd.read_csv(csv_file)
        self.max_length = max_length  # Set max length for padding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the row by index
        review = self.data.loc[idx, 'review']
        # assuming tokenized is stored as a string of list
        tokenized = eval(self.data.loc[idx, 'tokenized'])
        label = self.data.loc[idx, 'label']

        # Convert to tensor and pad
        tokenized_tensor = torch.tensor(tokenized, dtype=torch.long)
        tokenized_tensor = F.pad(
            tokenized_tensor, (0, self.max_length - len(tokenized_tensor)), value=0
        )  # Pad with zeros up to max_length

        label_tensor = torch.tensor(label, dtype=torch.long)

        return tokenized_tensor, label_tensor


# Paths to CSV files
train_csv = 'imdb_train.csv'
test_csv = 'imdb_test.csv'

# Load datasets
train_dataset = SentimentAnalysisDataset(train_csv)
test_dataset = SentimentAnalysisDataset(test_csv)

# Combine train and test datasets
combined_dataset = ConcatDataset([train_dataset, test_dataset])

# Define split proportions
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Calculate dataset sizes
train_size = int(train_ratio * len(combined_dataset))
val_size = int(val_ratio * len(combined_dataset))
test_size = len(combined_dataset) - train_size - val_size

# Split combined dataset
new_train_dataset, new_val_dataset, new_test_dataset = random_split(
    combined_dataset, [train_size, val_size, test_size]
)

# Create data loaders
batch_size = 32
train_loader = DataLoader(
    new_train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(new_val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(
    new_test_dataset, batch_size=batch_size, shuffle=False)

# Define the RNN model for sentiment analysis


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(EncoderRNN, self).__init__()
        """
        N: num batches (sentences).
        D: Each token is represented by a D-dimensional embedding vector (embed_size).
        T: Maximum sequence length (number of words in each sequence)
        H: hidden_size

        shape input: (N, T)
        shape embeded input: (N, T, D)
        
        for each word in the sentence:
            next_h = torch.tanh(x.mm(Wx) + prev_h.mm(Wh) + b)
            {
                where:
                x:      (N,D)
                Wx:     (D,H)
                Wh:     (H,H)
                b:      (H,)
                next_h: (N,H)
            }
            (This is one step in the image above.)
        
        This process repeats for all the words in the sentence, so the number of output or hidden states at the end is T.

        output: (N, T, H)
        """
        self.embedding = nn.Embedding(
            vocab_size, embed_size)  # assigne a vector of embec_size to each word
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)

        self.hidden_dim = hidden_size

    def forward(self, x):
        """
        hidden: (N, H)
        """
        x = self.embedding(x)
        output, hidden = self.rnn(x)
        return hidden


class DecoderRNN(nn.Module):
    def __init__(self, output_dim, hidden_dim):
        super(DecoderRNN, self).__init__()
        self.rnn = nn.RNN(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.hidden_dim = hidden_dim

    def forward(self, hidden):
        batch_size = hidden.size(1)
        input = torch.zeros(batch_size, 1, self.hidden_dim).to(
            hidden.device)  # [batch_size, 1, hidden_dim]
        outputs, hidden = self.rnn(input, hidden)

        # Pass final RNN output to linear layer
        # prediction = [batch_size, output_dim]
        prediction = self.fc(outputs.squeeze(1))
        return prediction


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src):
        hidden = self.encoder(src)
        print(f'output hidden: f{hidden.shape}')
        output = self.decoder(hidden)
        print(f'final output : f{output.shape}')
        return output.reshape((-1,))


# Set parameters
vocab_size = 88585  # As per your dataset
embed_size = 128
hidden_size = 64
output_size = 1  # Assuming binary classification: positive or negative sentiment
num_epochs = 10
learning_rate = 0.001


# Instantiate the model, define the loss function and optimizer
encoder = EncoderRNN(vocab_size, embed_size, hidden_size)
decoder = DecoderRNN(output_size, hidden_size)
model = Seq2Seq(encoder, decoder).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training function


def train_model(model, train_loader, val_loader):
    print('********* train model **********')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation accuracy after each epoch
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.float().tolist())

        val_accuracy = accuracy_score(all_labels, all_preds)
        print(
            f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Val Accuracy: {val_accuracy:.4f}')

# Testing function


def evaluate_model(model, test_loader):
    print('********* evaluate model **********')
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.float().tolist())

    test_accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=[
                                   "Negative", "Positive"])
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print("Classification Report:\n", report)


# Train the model
train_model(model, train_loader, val_loader)

# Evaluate the model on test data
evaluate_model(model, test_loader)
