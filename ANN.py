import numpy as np



def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)



X = np.array([
    [2, 60],
    [4, 70],
    [6, 80],
    [8, 90],
    [1, 50],
    [9, 95]
])


y = np.array([[0], [0], [1], [1], [0], [1]])



X = X / np.max(X, axis=0)



np.random.seed(42)
input_neurons = 2
hidden_neurons = 4
output_neurons = 1
learning_rate = 0.1
epochs = 5000

W1 = np.random.randn(input_neurons, hidden_neurons)
b1 = np.zeros((1, hidden_neurons))

W2 = np.random.randn(hidden_neurons, output_neurons)
b2 = np.zeros((1, output_neurons))


# ============================
# Training Loop
# ============================
for epoch in range(epochs):

    # -------- Forward Propagation --------
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, W2) + b2
    y_pred = sigmoid(z2)

    # -------- Loss (Mean Squared Error) --------
    loss = np.mean((y - y_pred) ** 2)

    # -------- Backpropagation --------
    d_loss = y - y_pred
    d_output = d_loss * sigmoid_derivative(y_pred)

    d_hidden = np.dot(d_output, W2.T) * relu_derivative(z1)

    # -------- Update Weights --------
    W2 += np.dot(a1.T, d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    W1 += np.dot(X.T, d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")


# ============================
# Final Prediction
# ============================
print("\nFinal Predictions:")
print(y_pred.round(3))
