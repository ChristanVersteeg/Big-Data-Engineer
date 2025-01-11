import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import re
import col
import data

df = data.df
df['Sentiment'] = np.where(df[col.NEGATIVE_REVIEW].str.strip() == 'No Negative', 1, 0)
df['Review'] = df[col.POSITIVE_REVIEW] + " " + df[col.NEGATIVE_REVIEW]
df = df[['Review', 'Sentiment']].dropna()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['Review'] = df['Review'].apply(clean_text)

word_to_idx = {"<PAD>": 0, "<UNK>": 1}
idx_to_word = {0: "<PAD>", 1: "<UNK>"}

def build_vocab(reviews, max_vocab_size=10000):
    word_counts = {}
    for review in reviews:
        for word in review.split():
            word_counts[word] = word_counts.get(word, 0) + 1
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    for idx, (word, _) in enumerate(sorted_words[:max_vocab_size - 2], start=2):
        word_to_idx[word] = idx
        idx_to_word[idx] = word

build_vocab(df['Review'])

def tokenize(text):
    return [word_to_idx.get(word, word_to_idx["<UNK>"]) for word in text.split()]

df['Tokenized'] = df['Review'].apply(tokenize)

def pad_sequence(tokens, max_len):
    if len(tokens) < max_len:
        return tokens + [word_to_idx["<PAD>"]] * (max_len - len(tokens))
    else:
        return tokens[:max_len]

MAX_LEN = 100
df['Padded'] = df['Tokenized'].apply(lambda x: pad_sequence(x, MAX_LEN))

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

class SentimentDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        review = torch.tensor(self.data.iloc[idx]['Padded'], dtype=torch.long)
        sentiment = torch.tensor(self.data.iloc[idx]['Sentiment'], dtype=torch.float)
        return review, sentiment

train_dataset = SentimentDataset(train_data)
test_dataset = SentimentDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=word_to_idx["<PAD>"])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        output = self.fc(hidden[-1])
        return self.sigmoid(output)

VOCAB_SIZE = len(word_to_idx)
EMBED_DIM = 64
HIDDEN_DIM = 128
OUTPUT_DIM = 1

model = SentimentModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_epoch(model, data_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for reviews, sentiments in data_loader:
        reviews, sentiments = reviews.to(device), sentiments.to(device)
        optimizer.zero_grad()
        predictions = model(reviews).squeeze()
        loss = criterion(predictions, sentiments)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for reviews, sentiments in data_loader:
            reviews, sentiments = reviews.to(device), sentiments.to(device)
            predictions = model(reviews).squeeze()
            loss = criterion(predictions, sentiments)
            total_loss += loss.item()
            correct += ((predictions > 0.5) == sentiments).sum().item()
    accuracy = correct / len(data_loader.dataset)
    return total_loss / len(data_loader), accuracy

NUM_EPOCHS = 5
for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_accuracy = evaluate_model(model, test_loader, criterion)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

def predict_sentiment(text):
    model.eval()
    with torch.no_grad():
        tokens = tokenize(clean_text(text))
        padded_tokens = pad_sequence(tokens, MAX_LEN)
        input_tensor = torch.tensor(padded_tokens, dtype=torch.long).unsqueeze(0).to(device)
        prediction = model(input_tensor).item()
        return "Positive" if prediction > 0.5 else "Negative"

print(predict_sentiment("The hotel was fantastic!"))
print(predict_sentiment("The service was horrible."))