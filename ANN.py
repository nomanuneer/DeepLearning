import numpy as np
import pandas as pd

# Simple binary classification dataset
X = np.array([
    [2, 60],
    [4, 70],
    [6, 80],
    [8, 90],
    [1, 50],
    [9, 95]
])

y = np.array([[0], [0], [1], [1], [0], [1]])

# Normalize features
X = X / np.max(X, axis=0)



class ANNFromScratch:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.1):
        np.random.seed(42)
        self.lr = lr
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

    # Activation functions
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.sigmoid(self.z2)

    def backward(self, X, y, y_pred):
        error = y - y_pred
        d_out = error * self.sigmoid_derivative(y_pred)

        d_hidden = np.dot(d_out, self.W2.T) * self.relu_derivative(self.z1)

        self.W2 += np.dot(self.a1.T, d_out) * self.lr
        self.b2 += np.sum(d_out, axis=0, keepdims=True) * self.lr

        self.W1 += np.dot(X.T, d_hidden) * self.lr
        self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * self.lr

    def train(self, X, y, epochs=5000):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = np.mean((y - y_pred) ** 2)
            self.backward(X, y, y_pred)

            if epoch % 1000 == 0:
                print(f"[Scratch] Epoch {epoch} | Loss: {loss:.4f}")


scratch_model = ANNFromScratch(input_dim=2, hidden_dim=4, output_dim=1)
scratch_model.train(X, y)

scratch_preds = scratch_model.forward(X)
print("\nScratch Model Predictions:")
print(np.round(scratch_preds, 3))



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

keras_model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

keras_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

keras_model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=8,
    verbose=0
)

loss, acc = keras_model.evaluate(X_test, y_test, verbose=0)
print(f"\n[Keras] Test Accuracy: {acc:.2f}")
