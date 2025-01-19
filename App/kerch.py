import re
import numpy as np

import data
import col

def isolate_sentiment_columns():
    df = data.df

    df['Sentiment'] = np.where(df[col.NEGATIVE_REVIEW].str.strip() == 'No Negative', 1, 0)
    df['Review'] = df[col.POSITIVE_REVIEW].fillna('') + ' ' + df[col.NEGATIVE_REVIEW].fillna('')
    df = df[['Review', 'Sentiment']].dropna()
    
    return df

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text