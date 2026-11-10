import numpy as np
import matplotlib.pyplot as plt
class NeuralNetwork:
    def __init__(self, vocab_size, block_size=3, embedding_dim=32, hidden_dim=64):
        self.D = embedding_dim
        self.H = hidden_dim
        self.T = block_size
        rng = np.random.default_rng(42)
        # Character and positional embeddings
        self.C = rng.normal(0, 0.3, (vocab_size, self.D))
        self.P = rng.normal(0, 0.05, (self.T, self.D))
        # Attention projections
        self.W_Q = rng.normal(0, 1/np.sqrt(self.D), (self.D, self.D))
        self.W_K = rng.normal(0, 1/np.sqrt(self.D), (self.D, self.D))
        self.W_V = rng.normal(0, 1/np.sqrt(self.D), (self.D, self.D))
        # Feed-forward layer
        self.W1 = rng.normal(0, np.sqrt(2/(2*self.D+self.H)),
                             (2*self.D, self.H))
        self.b1 = np.zeros((1, self.H))
        # Output classifier
        self.W2 = rng.normal(0, 1/np.sqrt(self.H),
                             (self.H, vocab_size))
        self.b2 = np.zeros((1, vocab_size))
    # --------------------------------------------------------
    # SOFTMAX
    # --------------------------------------------------------
    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    # --------------------------------------------------------
    # FORWARD PASS
    # --------------------------------------------------------
    def forward(self, X, cache=False):
        # X: (B, T), character indices
        # E: (B, T, D), embeddings + positional embeddings
        E = self.C[X] + self.P
        # Query comes from the last character
        Q = E[:, -1, :] @ self.W_Q
        # Keys and values come from all characters
        K = E @ self.W_K
        V = E @ self.W_V
        # Scaled dot-product attention
        scores = np.sum(Q[:, None, :] * K, axis=2) / np.sqrt(self.D)
        wei = self.softmax(scores)
        # Attention output
        Xj = np.sum(wei[:, :, None] * V, axis=1)
        # Concatenate attention output and last embedding
        Xh = np.concatenate([Xj, E[:, -1, :]], axis=1)
        # Feed-forward layer
        H = np.tanh(Xh @ self.W1 + self.b1)
        # Output layer
        logits = H @ self.W2 + self.b2
        if cache:
            return logits, (X, E, Q, K, V, wei, Xh, H)
        return logits
    # --------------------------------------------------------
    # CROSS ENTROPY
    # --------------------------------------------------------
    def cross_entropy(self, logits, Y):
        m = len(Y)
        maximum = np.max(logits, axis=1, keepdims=True)
        log_sum = maximum[:, 0] + np.log(
            np.sum(np.exp(logits - maximum), axis=1)
        )
        return np.mean(log_sum - logits[np.arange(m), Y])
    # --------------------------------------------------------
    # BACKPROPAGATION
    # --------------------------------------------------------
    def backward(self, logits, Y, c):
        X, E, Q, K, V, wei, Xh, H = c
        m = len(Y)
        # Output softmax + cross entropy
        d_out = self.softmax(logits)
        d_out[np.arange(m), Y] -= 1
        d_out /= m
        g_W2 = H.T @ d_out
        g_b2 = np.sum(d_out, axis=0, keepdims=True)
        d_H = d_out @ self.W2.T
        # Backpropagation through tanh
        d_H *= (1 - H**2)
        g_W1 = Xh.T @ d_H
        g_b1 = np.sum(d_H, axis=0, keepdims=True)
        d_Xh = d_H @ self.W1.T
        # Split gradients of concatenated representation
        d_Xj = d_Xh[:, :self.D]
        d_last = d_Xh[:, self.D:]
        # Xj = sum(wei * V)
        d_V = wei[:, :, None] * d_Xj[:, None, :]
        d_wei = np.sum(
            d_Xj[:, None, :] * V, axis=2
        )
        # Backpropagation through attention softmax
        d_scores = wei * (
            d_wei - np.sum(d_wei * wei, axis=1, keepdims=True)
        )
        # scores = Q * K / sqrt(D)
        scale = np.sqrt(self.D)
        d_Q = np.sum(
            d_scores[:, :, None] * K, axis=1
        ) / scale
        d_K = d_scores[:, :, None] * Q[:, None, :] / scale
        # Gradients of attention projections
        g_W_Q = E[:, -1, :].T @ d_Q
        g_W_K = E.reshape(-1, self.D).T @ d_K.reshape(-1, self.D)
        g_W_V = E.reshape(-1, self.D).T @ d_V.reshape(-1, self.D)
        # Gradients towards embeddings
        d_E = d_K @ self.W_K.T + d_V @ self.W_V.T
        d_E[:, -1, :] += d_Q @ self.W_Q.T + d_last
        # Character embedding gradients
        g_C = np.zeros_like(self.C)
        np.add.at(g_C, X, d_E)
        # Positional embedding gradients
        g_P = np.sum(d_E, axis=0)
        return {
            "C": g_C, "P": g_P,
            "W_Q": g_W_Q, "W_K": g_W_K, "W_V": g_W_V,
            "W1": g_W1, "b1": g_b1,
            "W2": g_W2, "b2": g_b2
        }
    # --------------------------------------------------------
    # GRADIENT DESCENT
    # --------------------------------------------------------
    def gradient_descent(self, X, Y, iterations, batch_size=100):
        error = []
        rng = np.random.default_rng(123)
        for i in range(iterations):
            # Random mini-batch
            idx = rng.choice(
                len(X), min(batch_size, len(X)), replace=False
            )
            Xi = X[idx]
            Yi = Y[idx]
            # Forward pass
            logits, c = self.forward(Xi, cache=True)
            loss = self.cross_entropy(logits, Yi)
            # Backpropagation
            grads = self.backward(logits, Yi, c)
            # Gradient clipping
            norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
            if norm > 1.0:
                grads = {k: g / norm for k, g in grads.items()}
            # Learning rate decay
            lr = 0.05 * (1 - 0.9 * i / iterations)
            # Update parameters using gradient descent
            self.C -= lr * grads["C"]
            self.P -= lr * grads["P"]
            self.W_Q -= lr * grads["W_Q"]
            self.W_K -= lr * grads["W_K"]
            self.W_V -= lr * grads["W_V"]
            self.W1 -= lr * grads["W1"]
            self.b1 -= lr * grads["b1"]
            self.W2 -= lr * grads["W2"]
            self.b2 -= lr * grads["b2"]
            error.append(loss)
            if (i + 1) % 500 == 0:
                print(f"Iteration {i+1}: Loss = {loss:.4f}")
        return error
    # --------------------------------------------------------
    # GENERATE NAMES
    # --------------------------------------------------------
    def generate_name(self, itos, temperature=0.8):
        context = [0] * self.T
        name = ""
        for i in range(20):
            X = np.array([context])
            logits = self.forward(X)
            # Temperature sampling
            probs = self.softmax(logits / temperature)[0]
            # Avoid ending the name too early
            if len(name) < 2:
                probs[0] = 0
                probs /= np.sum(probs)
            sample = np.random.choice(
                len(probs), p=probs
            )
            # End token
            if sample == 0:
                break
            name += itos[sample]
            # Slide context window
            context = context[1:] + [sample]
        return name
# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Read dataset
    words = open("names.txt", "r").read().splitlines()
    # Vocabulary
    chars = sorted(set("".join(words)))
    stoi = {s: i+1 for i, s in enumerate(chars)}
    stoi["."] = 0
    itos = {i: s for s, i in stoi.items()}
    vocab_size = len(stoi)
    # Build dataset
    block_size = 3
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context.copy())
            Y.append(ix)
            context = context[1:] + [ix]
    X = np.array(X, dtype=np.int64)
    Y = np.array(Y, dtype=np.int64)
    # Initialize network
    neural_network = NeuralNetwork(
        vocab_size=vocab_size,
        block_size=block_size,
        embedding_dim=32,
        hidden_dim=64
    )
    # Training
    error = neural_network.gradient_descent(
        X, Y, iterations=10000, batch_size=100
    )
    # Generate names
    print("\n--- Generated names ---")
    for i in range(20):
        name = neural_network.generate_name(
            itos, temperature=0.8
        )
        print(name)
    # Plot loss
    plt.plot(error)
    plt.ylabel("Loss")
    plt.xlabel("Iterations")
    plt.title("Training Loss")
    plt.show()
