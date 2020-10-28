import numpy as np
import matplotlib.pyplot as plt
#import pylab as plt

class NeuralNetwork:
    def __init__(self):
        #np.random.seed(10) # for generating the same results
        self.wij   = np.random.rand(2,2) # input to hidden layer weights
        self.bj    = np.random.rand(2,1) # bias input
        self.wjk   = np.random.rand(2,1) # hidden layer to output weights
        self.bk    = np.random.rand(1,1) # bias output
        
    def sigmoid(self, x, w, b):
        z = np.dot(x, w) + b.T
        return 1/(1 + np.exp(-z))
    
    def sigmoid_derivative(self, x, w, b):
        return self.sigmoid(x, w, b) * (1 - self.sigmoid(x, w, b))
    
    def gradient_descent(self, x, y, iterations):
        for i in range(iterations):
            Xi = x
            Xj = self.sigmoid(Xi, self.wij, self.bj)
            yhat = self.sigmoid(Xj, self.wjk, self.bk)
            # gradients for hidden to output weights
            g_wjk = np.dot(Xj.T, (y - yhat) * self.sigmoid_derivative(Xj, self.wjk, self.bk))
            # gradients for input to hidden weights
            #los gradientes(derivadas) son la razón de cambio del error con respecto a los pesos por lo tanto si la razón de cambio es positiva significa
            #que la pendiente de la recta es positiva por lo tanto el error va en aumento y hay que restar el gradiente de los pesos.
            g_wij = np.dot(Xi.T, np.dot((y - yhat) * self.sigmoid_derivative(Xj, self.wjk, self.bk), self.wjk.T) * self.sigmoid_derivative(Xi, self.wij, self.bj))
            # gradients for input to hidden bias
            #g_bk = np.sum(np.dot(1, (y - yhat) * self.sigmoid_derivative(Xj, self.wjk, self.bk)).T, axis=1, keepdims=True)
            g_bk = np.sum(((y - yhat) * self.sigmoid_derivative(Xj, self.wjk, self.bk)).T, axis=1, keepdims=True)
            #g_bj = np.sum(np.dot(1, np.dot((y - yhat) * self.sigmoid_derivative(Xj, self.wjk, self.bk), self.wjk.T) * self.sigmoid_derivative(Xi, self.wij, self.bj)).T, axis=1, keepdims=True)
            g_bj = np.sum((np.dot((y - yhat) * self.sigmoid_derivative(Xj, self.wjk, self.bk), self.wjk.T) * self.sigmoid_derivative(Xi, self.wij, self.bj)).T, axis=1, keepdims=True)
            # update weights and bias we sum the gradients because the MSE(error cuadrático medio ) is calculated like this (yhat-y) and we are using (y-yhat) so there is the change of sign
            self.wij += g_wij
            self.wjk += g_wjk
            self.bj  += g_bj
            self.bk  += g_bk           
            error.extend([y[0]-yhat[0]])
        print('The final prediction from neural network are: ')
        print(yhat)

if __name__ == '__main__':
    neural_network = NeuralNetwork()
    print('Random starting input to hidden weights: ')
    print(neural_network.wij)
    print('Random starting hidden to output weights: ')
    print(neural_network.wjk)
    X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    y = np.array([[0, 1, 1, 0]]).T
    error = []
    testa = []
    testb = []
    testc = []
    pretesta = []
    pretestb = []
    pretestc = []
    neural_network.gradient_descent(X, y, 1000)
    print('Final input to hidden weights: ')
    print(neural_network.wij)
    print('Final hidden to output weights: ')
    print(neural_network.wjk)
    print('Final input to hidden bias: ')
    print(neural_network.bj)
    print('Final hidden to output bias: ')
    print(neural_network.bk)
    for i in range(0,100):
        pretesta = []
        pretestb = []
        pretestc = []
        for j in range(0,100):
            x1 = i/100
            x2 = j/100
            #Xi = [x1, x2]
            Xi = np.array([x1, x2])
            Xj = neural_network.sigmoid(Xi, neural_network.wij, neural_network.bj)
            #print(Xj[0])
            #print(Xj)
            #print(Xj[0][1])
            yhat = neural_network.sigmoid(Xj, neural_network.wjk, neural_network.bk)
            pretesta.extend([Xj[0][0]])
            pretestb.extend([Xj[0][1]])
            pretestc.extend(yhat[0])
        testa.extend([pretesta])
        testb.extend([pretestb])
        testc.extend([pretestc])
    plt.imshow(testa, cmap='gray', interpolation='nearest')   #plotear la segmentación de la primera neurona(activacion)
    plt.show()
    plt.imshow(testb, cmap='gray', interpolation='nearest')   #plotear la segmentación de la segunda neurona(activacion)
    plt.show()
    plt.imshow(testc, cmap='gray', interpolation='nearest')   #plotear la segmentacion en la salida(activación)
    plt.show()
    plt.plot(error)
    plt.ylabel('some numbers')
    plt.show()
