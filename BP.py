import numpy as np

# Configuration
input_dim, hidden_dim, output_dim = 2, 4, 1
learning_rate, epochs = 0.1, 10000
np.random.seed(42)

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights
W1 = np.random.randn(input_dim, hidden_dim) * 0.1
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim) * 0.1
b2 = np.zeros((1, output_dim))

# Training data (XOR)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Training loop
for epoch in range(epochs):
    # Forward pass
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    
    # Backward pass
    loss_grad = a2 - y
    d_output = loss_grad * sigmoid_derivative(a2)
    
    d_W2 = a1.T @ d_output
    d_b2 = np.sum(d_output, axis=0, keepdims=True)
    
    d_hidden = d_output @ W2.T * sigmoid_derivative(a1)
    
    d_W1 = X.T @ d_hidden
    d_b1 = np.sum(d_hidden, axis=0, keepdims=True)
    
    # Update weights
    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1
    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2
    
    if epoch % 1000 == 0:
        loss = np.mean((a2 - y) ** 2)
        print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

# Predictions
z1 = X @ W1 + b1
a1 = sigmoid(z1)
z2 = a1 @ W2 + b2
predictions = sigmoid(z2)

print("\nFinal Predictions:")
print(np.round(predictions, 3))
