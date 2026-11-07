import numpy as np
import math
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self):
        # Pesos de proyección para el bloque de Atención (Reemplazan a wij y bj)
        # Dimensión de entrada = 6, Dimensión oculta/Atención = 100
        self.W_Q = np.random.uniform(low=-0.1, high=0.1, size=(6, 100))
        self.W_K = np.random.uniform(low=-0.1, high=0.1, size=(6, 100))
        self.W_V = np.random.uniform(low=-0.1, high=0.1, size=(6, 100))
        
        # Pesos de la capa de salida (Clasificador)
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
        for i in range(iterations):
            # Muestreo de mini-batches (Tamaño = 100)
            idx = np.random.choice(np.arange(len(x)), 100, replace=False)
            Xi = x[idx]
            Yi = y[idx]
            
            # --- FORWARD PASS (Capa de Self-Attention) ---
            Q = np.dot(Xi, self.W_Q)                       # (100, 100)
            K = np.dot(Xi, self.W_K)                       # (100, 100)
            V = np.dot(Xi, self.W_V)                       # (100, 100)
            
            # Puntuaciones y pesos de atención (wei)
            scores = np.dot(Q, K.T)                        # (100, 100)
            exps = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            wei = exps / np.sum(exps, axis=1, keepdims=True) # (100, 100)
            
            # Representación contextualizada (Xj)
            Xj = np.dot(wei, V)                            # (100, 100)
            
            # Capa lineal de salida + Softmax
            yhat = self.stable_softmax(Xj, self.wjk, self.bk)
            yhat = np.clip(yhat, 1e-7, 1 - 1e-7)
            
            # --- BACKPROPAGATION ---
            # Gradiente de la capa de salida
            d_out = self.delta_cross_entropy(yhat, Yi)
            g_wjk = np.dot(Xj.T, d_out)
            g_bk  = np.sum(d_out, axis=0, keepdims=True).T
            
            # Gradiente hacia el estado oculto Xj
            d_Xj = np.dot(d_out, self.wjk.T)               # (100, 100)
            
            # Gradiente a través del cálculo de Xj = wei * V
            d_V = np.dot(wei.T, d_Xj)                      # (100, 100)
            d_wei = np.dot(d_Xj, V.T)                      # (100, 100)
            
            # Gradiente del Softmax de Atención
            d_scores = wei * (d_wei - np.sum(d_wei * wei, axis=1, keepdims=True))
            
            # Gradiente de la multiplicación de matrices scores = Q * K^T
            d_Q = np.dot(d_scores, K)                      # (100, 100)
            d_K = np.dot(d_scores.T, Q)                    # (100, 100)
            
            # Gradientes finales de los proyectores Q, K, V
            g_W_Q = np.dot(Xi.T, d_Q)                      # (6, 100)
            g_W_K = np.dot(Xi.T, d_K)                      # (6, 100)
            g_W_V = np.dot(Xi.T, d_V)                      # (6, 100)
            
            # --- ACTUALIZACIÓN DE PARÁMETROS ---
            learning_rate = 1.0
            self.W_Q -= g_W_Q * learning_rate
            self.W_K -= g_W_K * learning_rate
            self.W_V -= g_W_V * learning_rate
            self.wjk -= g_wjk * learning_rate
            self.bk  -= g_bk  * learning_rate
            
            # Guardar la pérdida (loss) global
            loss = self.cross_entropy(yhat, Yi)
            error.append(loss)
            
        np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
        self.yhat = yhat
        self.y = Yi


if __name__ == '__main__':
    neural_network = NeuralNetwork()
    
    # Lectura del dataset de nombres
    words = open('names.txt', 'r').read().splitlines()

    # Construcción del vocabulario e índices de mapeo
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}

    # Construcción del dataset estructurado con block_size = 3
    block_size = 3 
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix] 
    
    n_values = 27
    Y = np.eye(n_values)[Y]
    
    # Matriz de embeddings para caracteres
    C = np.random.uniform(low=-1, high=1, size=(27, 2))
    emb = C[X]
    emb = np.reshape(emb, (len(X), 6))

    X_train = emb
    y_train = Y
    error = []
    
    # Entrenamiento mediante descenso de gradiente
    neural_network.gradient_descent(X_train, y_train, 2500)
    
    # --- MUESTREO / GENERACIÓN DE NUEVOS NOMBRES ---
    letters = [".", "a", "b","c", "d", "e","f", "g", "h","i", "j", "k","l", "m", "n","o", "p", "q","r", "s", "t","u","v","w", "x", "y","z"]
    print("--- Nombres generados ---")
    for u in range(10):
        name = ""
        init = C[0]
        init = np.resize(init, (1, 6))
        
        for i in range(25):
            # Forward pass de atención para un único elemento
            Q = np.dot(init, neural_network.W_Q)
            K = np.dot(init, neural_network.W_K)
            V = np.dot(init, neural_network.W_V)
            
            scores = np.dot(Q, K.T)
            exps = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            wei = exps / np.sum(exps, axis=1, keepdims=True)
            Xj = np.dot(wei, V)
            
            predic = neural_network.stable_softmax(Xj, neural_network.wjk, neural_network.bk)
            
            sample = np.random.choice(np.arange(0, 27, 1), 1, p=predic[0])
            if i == 0:
                sample = np.random.randint(1, 26)
                init[0][0:2] = init[0][2:4] 
                init[0][2:4] = init[0][4:] 
                init[0][4:] = C[sample]
                name = name + letters[sample]
            elif sample[0] == 0:
                break
            else:
                init[0][0:2] = init[0][2:4] 
                init[0][2:4] = init[0][4:] 
                init[0][4:] = C[sample[0]]
                name = name + letters[sample[0]]
        print(name)
        
    # Graficar la curva de aprendizaje (Loss over time)
    plt.plot(error)
    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.show()
