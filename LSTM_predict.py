import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -----------------------------
# 1. Create simple sequence data
# -----------------------------
# Example: [1,2,3] -> 4, [2,3,4] -> 5
def create_dataset(data, time_steps=3):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Simple sequence
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

X, y = create_dataset(data)

# Reshape for LSTM [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# -----------------------------
# 2. Build LSTM Model
# -----------------------------
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(3, 1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# -----------------------------
# 3. Train Model
# -----------------------------
model.fit(X, y, epochs=200, verbose=0)

# -----------------------------
# 4. Test Prediction
# -----------------------------
test_input = np.array([8, 9, 10])
test_input = test_input.reshape((1, 3, 1))

prediction = model.predict(test_input)

print("Predicted next number:", int(prediction[0][0]))
