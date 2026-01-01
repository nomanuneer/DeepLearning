import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


texts = [
    "I love this movie",
    "This product is amazing",
    "I am very happy",
    "I hate this",
    "This is terrible",
    "I am very sad"
]

labels = [1, 1, 1, 0, 0, 0]  # 1 = Positive, 0 = Negative


tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=5)


model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=16, input_length=5))
model.add(GRU(32))
model.add(Dense(1, activation='sigmoid'))


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


model.fit(
    padded_sequences,
    np.array(labels),
    epochs=30,
    verbose=1
)


test_text = ["I really love this product"]
test_seq = tokenizer.texts_to_sequences(test_text)
test_pad = pad_sequences(test_seq, maxlen=5)

prediction = model.predict(test_pad)

if prediction[0][0] > 0.5:
    print("Positive Sentiment 😊")
else:
    print("Negative Sentiment 😞")
