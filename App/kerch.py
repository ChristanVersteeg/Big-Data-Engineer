import re
import numpy as np

import data
import col

from sklearn.model_selection import train_test_split

PAD = "<PAD>"
UNK = "<UNK>"
MAX_VOCAB_SIZE = 10000
MAX_LEN = 100
EMBED_DIM = 64
HIDDEN_DIM = 128
EPOCHS = 5
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
POSITIVE = "Positive"
NEGATIVE = "Negative"

def isolate_sentiment_columns(panda):
    df = panda

    df['Sentiment'] = np.where(df[col.NEGATIVE_REVIEW].str.strip() == 'No Negative', 1, 0)
    df['Review'] = df[col.POSITIVE_REVIEW].fillna('') + ' ' + df[col.NEGATIVE_REVIEW].fillna('')
    df = df[['Review', 'Sentiment']].dropna()
    
    return df

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def separate_training_testing(df):
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    train_reviews = train_data['Review'].tolist()
    train_labels = np.array(train_data['Sentiment'].tolist())
    test_reviews = test_data['Review'].tolist()
    test_labels = np.array(test_data['Sentiment'].tolist())
    return train_reviews, train_labels, test_reviews, test_labels