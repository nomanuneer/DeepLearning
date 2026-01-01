import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


np.random.seed(42)
tf.random.set_seed(42)


VOCAB_SIZE = 1000
MAX_LEN = 6
EMBEDDING_DIM = 16
GRU_UNITS = 32
EPOCHS = 30


def load_data():
    texts = [
        "I love this movie",
        "This product is amazing",
        "I am very happy",
        "I hate this",
        "This is terrible",
        "I am very sad"
    ]
    labels = [1, 1, 1, 0, 0, 0]
    return texts, np.array(labels)


def preprocess_text(texts):
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_LEN)

    return padded, tokenizer


def build_model():
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        GRU(GRU_UNITS),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train():
    texts, labels = load_data()
    X, tokenizer = preprocess_text(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )

    model = build_model()
    model.summary()

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=1
    )

    return model, tokenizer


def predict_sentiment(model, tokenizer, text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)

    prediction = model.predict(padded)[0][0]
    return "Positive " if prediction > 0.5 else "Negative "


if __name__ == "__main__":
    model, tokenizer = train()

    test_sentence = "I really love this product"
    result = predict_sentiment(model, tokenizer, test_sentence)

    print(f"Sentence: {test_sentence}")
    print(f"Prediction: {result}")
