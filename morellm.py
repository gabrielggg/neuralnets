import numpy as np
import math
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self):
        # Attention projection weights (replacing wij and bj)
        # Input features = 6, Hidden/Attention dim = 100
        self.W_Q = np.random.uniform(low=-0.1, high=0.1, size=(6, 100))
        self.W_K = np.random.uniform(low=-0.1, high=0.1, size=(6, 100))
        self.W_V = np.random.uniform(low=-0.1, high=0.1, size=(6, 100))
        
        # Hidden layer to output weights
        self.wjk = np.random.uniform(low=-0.1, high=0.1, size=(100, 27))
        self.bk  = np.random.uniform(low=-0.1, high=0.1, size=(27, 1))
        
    def stable_softmax(self, x, w, b):
        X = np.dot(x, w) + b.T
        exps = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exps / exps.sum(axis=1, keepdims=True)

    def delta_cross_entropy(self, yhaty, yy):
        yhat_tmp = np.copy(yhaty)
        y_tmp = np.copy(yy).argmax(axis=1)
        m = y_tmp.shape[0]
        grad = yhat_tmp
        grad[range(m), y_tmp] -= 1
        grad = grad / m
        return grad

    def cross_entropy(self, yhaty, yy):
        yhat_tmp = np.copy(yhaty)
        y_tmp = np.copy(yy).argmax(axis=1)
        m = y_tmp.shape[0]
        p = yhat_tmp
        log_likelihood = -np.log(p[range(m), y_tmp])
        loss = np.sum(log_likelihood) / m
        return loss

    def gradient_descent(self, x, y, iterations):
        error = []
        for i in range(iterations):
            # Mini-batch sampling
            idx = np.random.choice(np.arange(len(x)), 100, replace=False)
            Xi = x[idx]
            Yi = y[idx]
            
            # --- FORWARD PASS (Self-Attention Layer) ---
            Q = np.dot(Xi, self.W_Q)                       # (100, 100)
            K = np.dot(Xi, self.W_K)                       # (100, 100)
            V = np.dot(Xi, self.W_V)                       # (100, 100)
            
            # Attention scores and weights (wei)
            scores = np.dot(Q, K.T)                        # (100, 100)
            exps = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            wei = exps / np.sum(exps, axis=1, keepdims=True) # (100, 100)
            
            # Weighted context layer output (Xj)
            Xj = np.dot(wei, V)                            # (100, 100)
            
            # Output classifier layer
            yhat = self.stable_softmax(Xj, self.wjk, self.bk)
            yhat = np.clip(yhat, 1e-7, 1 - 1e-7)
            
            # --- BACKPROPAGATION ---
            # Output layer gradient
            d_out = self.delta_cross_entropy(yhat, Yi)
            g_wjk = np.dot(Xj.T, d_out)
            g_bk  = np.sum(d_out, axis=0, keepdims=True).T
            
            # Backprop gradient through output weights to hidden state Xj
            d_Xj = np.dot(d_out, self.wjk.T)               # (100, 100)
            
            # Backprop through attention output calculation: Xj = wei * V
            d_V = np.dot(wei.T, d_Xj)                      # (100, 100)
            d_wei = np.dot(d_Xj, V.T)                      # (100, 100)
            
            # Backprop through Attention Softmax
            d_scores = wei * (d_wei - np.sum(d_wei * wei, axis=1, keepdims=True))
            
            # Backprop through matrix multiplication: scores = Q * K^T
            d_Q = np.dot(d_scores, K)                      # (100, 100)
            d_K = np.dot(d_scores.T, Q)                    # (100, 100)
            
            # Final parameter gradients for Q, K, V projections
            g_W_Q = np.dot(Xi.T, d_Q)                      # (6, 100)
            g_W_K = np.dot(Xi.T, d_K)                      # (6, 100)
            g_W_V = np.dot(Xi.T, d_V)                      # (6, 100)
            
            # --- UPDATE WEIGHTS ---
            learning_rate = 1.0
            self.W_Q -= g_W_Q * learning_rate
            self.W_K -= g_W_K * learning_rate
            self.W_V -= g_W_V * learning_rate
            self.wjk -= g_wjk * learning_rate
            self.bk  -= g_bk  * learning_rate
            
            # Tracking loss
            loss = self.cross_entropy(yhat, Yi)
            error.append(loss)
            
        self.yhat = yhat
        self.y = Yi
        return error
