"""
Trains a simple sentiment analysis model using Keras on hotel reviews.
Assumes `data.df` is a pandas DataFrame loaded elsewhere, and `col` contains
the column constants. Uses an LSTM-based model to classify reviews.
"""

import re
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras import layers

import data
import col

def isolate_sentiment_columns():
    df = data.df

    df['Sentiment'] = np.where(df[col.NEGATIVE_REVIEW].str.strip() == 'No Negative', 1, 0)
    df['Review'] = df[col.POSITIVE_REVIEW].fillna('') + ' ' + df[col.NEGATIVE_REVIEW].fillna('')
    df = df[['Review', 'Sentiment']].dropna()
    
    return df
df = isolate_sentiment_columns()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
df['Review'] = df['Review'].apply(clean_text)

def seperate_training_testing():
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    train_reviews = train_data['Review'].tolist()
    train_labels = np.array(train_data['Sentiment'].tolist())
    test_reviews = test_data['Review'].tolist()
    test_labels = np.array(test_data['Sentiment'].tolist())
    
    return train_reviews, train_labels, test_reviews, test_labels
train_reviews, train_labels, test_reviews, test_labels = seperate_training_testing()

def tokenize():
    max_vocab_size = 10000
    max_len = 100

    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<UNK>")
    tokenizer.fit_on_texts(train_reviews)

    train_sequences = tokenizer.texts_to_sequences(train_reviews)
    test_sequences = tokenizer.texts_to_sequences(test_reviews)

    train_padded = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
    test_padded  = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')
    return max_vocab_size, max_len, tokenizer, train_padded, test_padded
max_vocab_size, max_len, tokenizer, train_padded, test_padded = tokenize()

def build_model():
    embed_dim = 64
    hidden_dim = 128
    model = tf.keras.Sequential([
        layers.Embedding(input_dim=max_vocab_size, output_dim=embed_dim, input_length=max_len),
        layers.LSTM(hidden_dim),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.summary()
    return model
model = build_model()

def train_model():
    epochs = 5
    batch_size = 32

    model.fit(
        train_padded,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
train_model()

def prediction():
    test_loss, test_acc = model.evaluate(test_padded, test_labels, verbose=1)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    def predict_sentiment(text):
        text_cleaned = clean_text(text)
        seq = tokenizer.texts_to_sequences([text_cleaned])
        padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
        prediction = model.predict(padded)[0][0]
        return "Positive" if prediction > 0.5 else "Negative"

    print(predict_sentiment("The hotel was fantastic!"))
    print(predict_sentiment("The room was dirty and the service was terrible."))
prediction()