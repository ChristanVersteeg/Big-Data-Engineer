import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import kerch

df = kerch.isolate_sentiment_columns()
df['Review'] = df['Review'].apply(kerch.clean_text)

def separate_training_testing():
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    train_reviews = train_data['Review'].tolist()
    train_labels = np.array(train_data['Sentiment'].tolist())
    test_reviews = test_data['Review'].tolist()
    test_labels = np.array(test_data['Sentiment'].tolist())
    return train_reviews, train_labels, test_reviews, test_labels
train_reviews, train_labels, test_reviews, test_labels = separate_training_testing()

# ---------------------------------------
# 3. BUILD VOCABULARY, TOKENIZE, PAD
# ---------------------------------------
word_to_idx = {"<PAD>": 0, "<UNK>": 1}

def build_vocabulary(reviews, max_vocab_size=10000):
    word_counts = {}
    for review in reviews:
        for word in review.split():
            word_counts[word] = word_counts.get(word, 0) + 1
    # Sort by frequency (descending)
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    # Populate word_to_idx with top words
    idx = 2
    for word, _ in sorted_words[: max_vocab_size - 2]:
        word_to_idx[word] = idx
        idx += 1

def tokenize(text):
    return [word_to_idx.get(word, word_to_idx["<UNK>"]) for word in text.split()]

def pad_sequence(tokens, max_len):
    if len(tokens) < max_len:
        return tokens + [word_to_idx["<PAD>"]] * (max_len - len(tokens))
    else:
        return tokens[:max_len]

# Tokenization parameters
max_vocab_size = 10000
max_len = 100

# Build vocab on training reviews only
build_vocabulary(train_reviews, max_vocab_size=max_vocab_size)

# Convert reviews to padded sequences
def convert_to_padded_sequences(reviews, labels):
    tokenized = [tokenize(review) for review in reviews]
    padded = [pad_sequence(seq, max_len) for seq in tokenized]
    padded = np.array(padded, dtype=np.int64)
    labels = np.array(labels, dtype=np.float32)
    return padded, labels

train_padded, train_labels = convert_to_padded_sequences(train_reviews, train_labels)
test_padded, test_labels = convert_to_padded_sequences(test_reviews, test_labels)

# ---------------------------------------
# 4. DATASET & DATALOADER
# ---------------------------------------
class SentimentDataset(Dataset):
    def __init__(self, padded_data, labels):
        self.padded_data = padded_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        review = torch.tensor(self.padded_data[idx], dtype=torch.long)
        sentiment = torch.tensor(self.labels[idx], dtype=torch.float)
        return review, sentiment

train_dataset = SentimentDataset(train_padded, train_labels)
test_dataset = SentimentDataset(test_padded, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ---------------------------------------
# 5. BUILD THE MODEL
# ---------------------------------------
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embed_dim,
                                      padding_idx=word_to_idx["<PAD>"])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)            # (batch_size, seq_len, embed_dim)
        _, (hidden, _) = self.lstm(embedded)    # hidden shape: (num_layers, batch_size, hidden_dim)
        out = self.fc(hidden[-1])               # Take last layer's hidden
        return self.sigmoid(out)

vocab_size = len(word_to_idx)
embed_dim = 64
hidden_dim = 128
output_dim = 1

model = SentimentModel(vocab_size, embed_dim, hidden_dim, output_dim)
print(model)

# ---------------------------------------
# 6. TRAIN THE MODEL
# ---------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(epochs=5):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for reviews, sentiments in train_loader:
            reviews, sentiments = reviews.to(device), sentiments.to(device)
            optimizer.zero_grad()
            outputs = model(reviews).squeeze()
            loss = criterion(outputs, sentiments)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        val_loss, val_acc = evaluate_model()
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Accuracy: {val_acc:.4f}")

def evaluate_model():
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for reviews, sentiments in test_loader:
            reviews, sentiments = reviews.to(device), sentiments.to(device)
            outputs = model(reviews).squeeze()
            loss = criterion(outputs, sentiments)
            total_loss += loss.item()

            preds = (outputs > 0.5).float()
            correct += (preds == sentiments).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    return avg_loss, accuracy

train_model(epochs=5)

# ---------------------------------------
# 7. MAKE PREDICTIONS
# ---------------------------------------
def predict_sentiment(text):
    model.eval()
    with torch.no_grad():
        # Clean & convert text
        text_cleaned = kerch.clean_text(text)
        tokens = tokenize(text_cleaned)
        padded_tokens = pad_sequence(tokens, max_len)
        # Convert to tensor
        input_tensor = torch.tensor([padded_tokens], dtype=torch.long).to(device)
        output = model(input_tensor).item()
        return "Positive" if output > 0.5 else "Negative"

print("Prediction:", predict_sentiment("The hotel was fantastic!"))
print("Prediction:", predict_sentiment("The room was dirty and the service was terrible."))
