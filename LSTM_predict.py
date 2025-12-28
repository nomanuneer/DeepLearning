import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def create_sequences(data, time_steps=5):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)


data = np.arange(1, 51)

TIME_STEPS = 5


scaler = MinMaxScaler()
data = data.reshape(-1, 1)
scaled_data = scaler.fit_transform(data)

X, y = create_sequences(scaled_data, TIME_STEPS)

# Reshape for LSTM: [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -----------------------------
# 5. Build LSTM Model
# -----------------------------
model = Sequential()
model.add(LSTM(64, activation="tanh", input_shape=(TIME_STEPS, 1)))
model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mse"
)

# -----------------------------
# 6. Train Model
# -----------------------------
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=8,
    verbose=1
)

# -----------------------------
# 7. Evaluate Model
# -----------------------------
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

print("Test Mean Squared Error (MSE):", mse)

# -----------------------------
# 8. Predict Next Value
# -----------------------------
last_sequence = data[-TIME_STEPS:]
last_sequence = scaler.transform(last_sequence.reshape(-1, 1))
last_sequence = last_sequence.reshape((1, TIME_STEPS, 1))

next_value = model.predict(last_sequence)
next_value = scaler.inverse_transform(next_value)

print("Predicted next value:", int(next_value[0][0]))
