import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np
import os
import nltk
nltk.download('punkt')

# 1. Load and Combine the Data from Files with Debugging
def load_data(file_path):
    sentences = []
    labels = []
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return sentences, labels
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    sentence, label = parts
                    sentences.append(sentence)
                    labels.append(int(label))
            print(f"Loaded {len(sentences)} sentences from {file_path}. Sample data: {sentences[:3]}")
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
    
    return sentences, labels

# Load datasets
amazon_sentences, amazon_labels = load_data('amazon_cells_labelled.txt')
imdb_sentences, imdb_labels = load_data('imdb_labelled.txt')
yelp_sentences, yelp_labels = load_data('yelp_labelled.txt')

# Combine datasets
sentences = amazon_sentences + imdb_sentences + yelp_sentences
labels = amazon_labels + imdb_labels + yelp_labels

# Verify data loading was successful
if len(sentences) == 0 or len(labels) == 0:
    print("No data was loaded. Please check the file paths and ensure the files contain data.")
else:
    print(f"Total sentences loaded: {len(sentences)}")

# 2. Preprocess the Text
# Tokenize sentences and build vocabulary
all_words = [word for sentence in sentences for word in word_tokenize(sentence.lower())]
word_counts = Counter(all_words)
vocab = {word: idx+1 for idx, (word, _) in enumerate(word_counts.items())}  # +1 to reserve 0 for padding

# Convert sentences to sequences of word indices
def sentence_to_sequence(sentence):
    return [vocab.get(word, 0) for word in word_tokenize(sentence.lower())]

sequences = [sentence_to_sequence(sentence) for sentence in sentences]

# Pad sequences
sequence_length = 100  # Define a fixed length for input sequences
padded_sequences = np.zeros((len(sequences), sequence_length), dtype=int)
for i, seq in enumerate(sequences):
    padded_sequences[i, :min(len(seq), sequence_length)] = seq[:sequence_length]

# Verify padding results
print(f"Number of padded sequences: {len(padded_sequences)}")

# 3. Test Data Split with No Filter
# Use a small test size to retain more samples in the training set
try:
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.1, random_state=42)
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
except ValueError as e:
    print(f"Error during train-test split: {e}")

# Check if split worked
assert len(X_train) > 0, "Training set is empty after split."
assert len(X_test) > 0, "Test set is empty after split."

# 4. Define Custom Dataset
class SentimentDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = SentimentDataset(X_train, y_train)
test_dataset = SentimentDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Verify DataLoader sizes
print(f"Train loader batches: {len(train_loader)}, Test loader batches: {len(test_loader)}")

# 5. Define LSTM Model
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        h_n = h_n.squeeze(0)  # remove extra dimensions
        out = self.fc(h_n)
        return out

# Initialize model parameters
vocab_size = len(vocab) + 1  # +1 for padding index
embed_size = 100
hidden_size = 128
output_size = 2  # positive or negative

model = SentimentLSTM(vocab_size, embed_size, hidden_size, output_size)

# 6. Training Setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. Train the Model
epochs = 3  # reduce epochs for testing
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# 8. Evaluate the Model
model.eval()
y_pred, y_true = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        output = model(X_batch)
        _, predicted = torch.max(output, 1)
        y_pred.extend(predicted.tolist())
        y_true.extend(y_batch.tolist())

accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
