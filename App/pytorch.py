import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import data
import col

import kerch

df = kerch.isolate_sentiment_columns(data.df)
df['Review'] = df['Review'].apply(kerch.clean_text)
train_reviews, train_labels, test_reviews, test_labels = kerch.separate_training_testing(df)

word_to_idx = {kerch.PAD: 0, kerch.UNK: 1}

def build_vocabulary(reviews):
    word_counts = {}
    for text in reviews:
        for word in text.split():
            word_counts[word] = word_counts.get(word, 0) + 1

    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    idx = 2
    for word, _ in sorted_words[: kerch.MAX_VOCAB_SIZE - 2]:
        word_to_idx[word] = idx
        idx += 1

def tokenize(text):
    return [word_to_idx.get(word, word_to_idx[kerch.UNK]) for word in text.split()]

def pad_sequence(tokens):
    if len(tokens) < kerch.MAX_LEN:
        return tokens + [word_to_idx[kerch.PAD]] * (kerch.MAX_LEN - len(tokens))
    else:
        return tokens[:kerch.MAX_LEN]

def tokenization():
    build_vocabulary(train_reviews)

    train_sequences = [tokenize(review) for review in train_reviews]
    test_sequences  = [tokenize(review) for review in test_reviews]

    train_padded = [pad_sequence(seq) for seq in train_sequences]
    test_padded  = [pad_sequence(seq) for seq in test_sequences]

    train_padded = np.array(train_padded, dtype=np.int64)
    test_padded  = np.array(test_padded,  dtype=np.int64)

    return train_padded, test_padded
train_padded, test_padded = tokenization()

train_labels = np.array(train_labels, dtype=np.float32)
test_labels  = np.array(test_labels,  dtype=np.float32)

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embed_dim,
                                      padding_idx=word_to_idx[kerch.PAD])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)        
        _, (hidden, _) = self.lstm(embedded)
        out = self.fc(hidden[-1])           
        return self.sigmoid(out)

def build_model():
    output_dim = 1

    vocab_size = len(word_to_idx)
    model = SentimentModel(vocab_size, kerch.EMBED_DIM, kerch.HIDDEN_DIM, output_dim)
    print(model)
    return model
model = build_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

class SentimentDataset(Dataset):
    def __init__(self, padded_data, labels):
        self.padded_data = padded_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.padded_data[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float)
        return x, y

full_train_dataset = SentimentDataset(train_padded, train_labels)
test_dataset       = SentimentDataset(test_padded,  test_labels)

def train_model():

    dataset_size = len(full_train_dataset)
    val_size = int(kerch.VALIDATION_SPLIT * dataset_size)
    train_size = dataset_size - val_size

    train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=kerch.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_subset,   batch_size=kerch.BATCH_SIZE, shuffle=False)

    for epoch in range(1, kerch.EPOCHS + 1):
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

        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for reviews, sentiments in val_loader:
                reviews, sentiments = reviews.to(device), sentiments.to(device)
                outputs = model(reviews).squeeze()
                loss = criterion(outputs, sentiments)
                val_loss += loss.item()
                preds = (outputs > 0.5).float()
                correct += (preds == sentiments).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / len(val_subset)

        print(f"Epoch {epoch}/{kerch.EPOCHS}, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Accuracy: {val_acc:.4f}")

train_model()

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def prediction():
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

    avg_test_loss = total_loss / len(test_loader)
    test_acc = correct / len(test_loader.dataset)
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    def predict_sentiment(text):
        text_cleaned = kerch.clean_text(text)
        tokens = tokenize(text_cleaned)
        padded_tokens = pad_sequence(tokens)
        input_tensor = torch.tensor([padded_tokens], dtype=torch.long).to(device)
        with torch.no_grad():
            prediction = model(input_tensor).item()
        return prediction, "Positive" if prediction > 0.5 else "Negative"

    data.sample = kerch.isolate_sentiment_columns(data.sample)
    for i, row in data.sample.iterrows():
        score, sentiment = predict_sentiment(row['Review'])
        data.sample.at[i, col.REVIEWER_SCORE] = score * 10
        if sentiment == kerch.POSITIVE:
            data.sample.at[i, col.POSITIVE_REVIEW] = row['Review']
        else:
            data.sample.at[i, col.NEGATIVE_REVIEW] = row['Review']
    data.upload(data.sample, "Torch")
prediction()