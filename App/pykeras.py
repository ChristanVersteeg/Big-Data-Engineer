import dask.dataframe as dd
import tensorflow as tf
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras import layers
import data
import col
import kerch

df_dd = dd.from_pandas(data.df, npartitions=4)
df_dd[col.REVIEW] = df_dd[col.REVIEW].fillna('')
df_dd[col.REVIEW] = df_dd[col.REVIEW].apply(
    kerch.clean_text,
    meta=(col.REVIEW, 'object')
)
df = df_dd.compute()
df = kerch.isolate_sentiment_columns(df)

train_reviews, train_labels, test_reviews, test_labels = kerch.separate_training_testing(df)

def tokenize():
    tokenizer = Tokenizer(num_words=kerch.MAX_VOCAB_SIZE, oov_token=kerch.UNK)
    tokenizer.fit_on_texts(train_reviews)

    train_sequences = tokenizer.texts_to_sequences(train_reviews)
    test_sequences = tokenizer.texts_to_sequences(test_reviews)

    train_padded = pad_sequences(train_sequences, maxlen=kerch.MAX_LEN, padding='post', truncating='post')
    test_padded  = pad_sequences(test_sequences, maxlen=kerch.MAX_LEN, padding='post', truncating='post')
    return tokenizer, train_padded, test_padded
tokenizer, train_padded, test_padded = tokenize()

def build_model():
    model = tf.keras.Sequential([
        layers.Embedding(input_dim=kerch.MAX_VOCAB_SIZE, output_dim=kerch.EMBED_DIM, input_length=kerch.MAX_LEN),
        layers.LSTM(kerch.HIDDEN_DIM),
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
    model.fit(
        train_padded,
        train_labels,
        epochs=kerch.EPOCHS,
        batch_size=kerch.BATCH_SIZE,
        validation_split=kerch.VALIDATION_SPLIT,
        verbose=1
    )
train_model()

def prediction():
    test_loss, test_acc = model.evaluate(test_padded, test_labels, verbose=1)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    def predict_sentiment(text):
        text_cleaned = kerch.clean_text(text)
        seq = tokenizer.texts_to_sequences([text_cleaned])
        padded = pad_sequences(seq, maxlen=kerch.MAX_LEN, padding='post', truncating='post')
        prediction = model.predict(padded)[0][0]
        return prediction, kerch.POSITIVE if prediction > 0.5 else kerch.NEGATIVE

    data.sample = kerch.isolate_sentiment_columns(data.sample)
    for i, row in data.sample.iterrows():
        score, sentiment = predict_sentiment(row['Review'])
        data.sample.at[i, col.REVIEWER_SCORE] = score * 10
        if sentiment == kerch.POSITIVE:
            data.sample.at[i, col.POSITIVE_REVIEW] = row['Review']
        else:
            data.sample.at[i, col.NEGATIVE_REVIEW] = row['Review']
    data.upload(data.sample, "Keras")
prediction()