import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])


np.random.seed(42)
input_neurons = 2
hidden_neurons = 2
output_neurons = 1
learning_rate = 0.1
epochs = 10000

W1 = np.random.uniform(size=(input_neurons, hidden_neurons))
b1 = np.random.uniform(size=(1, hidden_neurons))

W2 = np.random.uniform(size=(hidden_neurons, output_neurons))
b2 = np.random.uniform(size=(1, output_neurons))

for epoch in range(epochs):

    # -------- Forward Propagation --------
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    predicted_output = sigmoid(final_input)

    # -------- Loss (MSE) --------
    loss = np.mean((y - predicted_output) ** 2)

    # -------- Backpropagation --------
    error = y - predicted_output
    d_predicted = error * sigmoid_derivative(predicted_output)

    error_hidden = d_predicted.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # -------- Update Weights --------
    W2 += hidden_output.T.dot(d_predicted) * learning_rate
    b2 += np.sum(d_predicted, axis=0, keepdims=True) * learning_rate

    W1 += X.T.dot(d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")

print("\nFinal Predictions:")
print(predicted_output.round(3))
