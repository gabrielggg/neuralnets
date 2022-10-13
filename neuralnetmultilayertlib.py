import numpy as np
import matplotlib.pyplot as plt
#import pylab as plt

class NeuralNetwork:
    def __init__(self):
        #np.random.seed(10) # for generating the same resultss
        
        self.wij   = np.random.rand(2,2) # input to hidden layer weights
        self.bj    = np.random.rand(2,1) # bias input
        self.wjk   = np.random.rand(2,2) # hidden layer to output weights
        self.bk    = np.random.rand(2,1) # bias output
        self.wkl   = np.random.rand(2,1) # hidden layer to output weights
        self.bl    = np.random.rand(1,1) # bias outputs
        self.wv = [self.wij, self.wjk, self.wkl]
        self.bv = [self.bj, self.bk, self.bl]
        
    def sigmoid(self, x, w, b):
        z = np.dot(x, w) + b.T
        return 1/(1 + np.exp(-z))
    
    def sigmoid_derivative(self, x, w, b):
        return self.sigmoid(x, w, b) * (1 - self.sigmoid(x, w, b))
    
    def gradient_descent(self, x, y, iterations):
        x_real = x
        for i in range(iterations):
            xv = []
            gwv = []
            gbv = []
            npdot = []
            npdotb = []
            ind = 0
            x = x_real
            for z in range (len(self.wv)):
                xv.append(x)
                #Xi = x
                print("**********************")
                print(x)
                print(xv[z],"a", self.wv[z],"b", self.bv[z], "c")
                print("**********************")
                x = self.sigmoid(xv[z], self.wv[z], self.bv[z])
                #Xj = self.sigmoid(Xi, self.wij, self.bj)
                if (z == len(self.wv) - 1):
                    yhat = self.sigmoid(xv[z], self.wv[z], self.bv[z])
                    # gradients for hidden to output weights
                    for z1 in range(len(self.wv) - 1, -1, -1):
                        if z1 == (len(self.wv) - 1) :
                            gwv.append(np.dot(xv[z1].T, (y - yhat) * self.sigmoid_derivative(xv[z1], self.wv[z1], self.bv[z1])))
                            gbv.append(np.sum(((y - yhat) * self.sigmoid_derivative(xv[z1], self.wv[z1], self.bv[z1])).T, axis=1, keepdims=True))
                            npdot.append(np.dot((y - yhat) * self.sigmoid_derivative(xv[z1], self.wv[z1], self.bv[z1]), self.wv[z1].T))
                            npdotb.append(np.dot((y - yhat) * self.sigmoid_derivative(xv[z1], self.wv[z1], self.bv[z1]), self.wv[z1].T))
                        else :
                            gwv.append(np.dot(xv[z1].T, npdot[ind] * self.sigmoid_derivative(xv[z1], self.wv[z1], self.bv[z1])))
                            gbv.append(np.sum((npdotb[ind] * self.sigmoid_derivative(xv[z1], self.wv[z1], self.bv[z1])).T, axis=1, keepdims=True))
                            npdot.append(np.dot( npdot[ind] * self.sigmoid_derivative(xv[z1], self.wv[z1], self.bv[z1]), self.wv[z1].T ))
                            npdotb.append(np.dot( npdotb[ind] * self.sigmoid_derivative(xv[z1], self.wv[z1], self.bv[z1]), self.wv[z1].T))
                            
                            ind += 1
                        #g_wjk = np.dot(Xj.T, (y - yhat) * self.sigmoid_derivative(Xj, self.wjk, self.bk))
                        
                        # gradients for input to hidden weights
                        #los gradientes(derivadas) son la razón de cambio del error con respecto a los pesos por lo tanto si la razón de cambio es positiva significa
                        #que la pendiente de la recta es positiva por lo tanto el error va en aumento y hay que restar el gradiente de los pesos.
                        #g_wkl = np.dot(Xk.T, (y - yhat) * self.sigmoid_derivative(Xk, self.wkl, self.bl))
                        #g_wjk = np.dot(Xj.T, np.dot((y - yhat) * self.sigmoid_derivative(Xl, self.wkl, self.bl), self.wkl.T) * self.sigmoid_derivative(Xj, self.wjk, self.bk))
                        #g_wij = np.dot(Xi.T, np.dot(np.dot((y - yhat) * self.sigmoid_derivative(Xk, self.wkl, self.bl), self.wkl.T) * self.sigmoid_derivative(Xj, self.wjk, self.bk), self.wjk.T) * self.sigmoid_derivative(Xi, self.wij, self.bj))
                        # gradients for input to hidden bias
                        #g_bk = np.sum(np.dot(1, (y - yhat) * self.sigmoid_derivative(Xj, self.wjk, self.bk)).T, axis=1, keepdims=True)
                        #g_bk = np.sum(((y - yhat) * self.sigmoid_derivative(Xj, self.wjk, self.bk)).T, axis=1, keepdims=True)
                        #g_bj = np.sum(np.dot(1, np.dot((y - yhat) * self.sigmoid_derivative(Xj, self.wjk, self.bk), self.wjk.T) * self.sigmoid_derivative(Xi, self.wij, self.bj)).T, axis=1, keepdims=True)
                        #g_bj = np.sum((np.dot((y - yhat) * self.sigmoid_derivative(Xj, self.wjk, self.bk), self.wjk.T) * self.sigmoid_derivative(Xi, self.wij, self.bj)).T, axis=1, keepdims=True)
                        # update weights and bias we sum the gradients because the MSE(error cuadrático medio ) is calculated like this (yhat-y) and we are using (y-yhat) so there is the change of sign
                    #self.wij += g_wij
                    #self.wjk += g_wjk
                    #self.bj  += g_bj
                    #self.bk  += g_bk
                    gwv = gwv[::-1]
                    gbv = gbv[::-1]
                    for zx in range (len(self.wv)): 
                        print("------------------------------------------")
                        print(self.wv[zx])
                        print("------------------------------------------")
                        print(gwv[zx])
                        self.wv[zx] += gwv[zx]  
                        self.bv[zx] += gbv[zx]        
                    error.extend([y[0]-yhat[0]])
        print('The final prediction from neural network are: ')
        print(yhat)
        print(gwv)

if __name__ == '__main__':
    neural_network = NeuralNetwork()
    print('Random starting input to hidden weights: ')
    print(neural_network.wv[0])
    print('Random starting hidden to output weights: ')
    print(neural_network.wv[1])
    X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    y = np.array([[0, 1, 1, 0]]).T
    error = []
    testa = []
    testb = []
    testa2 = []
    testb2 = []
    testc = []
    pretesta = []
    pretestb = []
    pretesta2 = []
    pretestb2 = []
    pretestc = []
    neural_network.gradient_descent(X, y, 10000)
    print('Final input to hidden weights: ')
    print(neural_network.wv[0])
    print('Final hidden to output weights: ')
    print(neural_network.wv[1])
    print('Final input to hidden bias: ')
    print(neural_network.bv[0])
    print('Final hidden to output bias: ')
    print(neural_network.bv[1])
    for i in range(0,100):
        pretesta = []
        pretestb = []
        pretesta2 = []
        pretestb2 = []
        pretestc = []
        for j in range(0,100):
            x1 = i/100
            x2 = j/100
            #Xi = [x1, x2]
            Xi = np.array([x1, x2])
            Xj = neural_network.sigmoid(Xi, neural_network.wv[0], neural_network.bv[0])
            Xk = neural_network.sigmoid(Xj, neural_network.wv[1], neural_network.bv[1])
            #print(Xj[0])
            #print(Xj)
            #print(Xj[0][1])
            yhat = neural_network.sigmoid(Xk, neural_network.wv[2], neural_network.bv[2])
            pretesta.extend([Xj[0][0]])
            pretestb.extend([Xj[0][1]])
            pretesta2.extend([Xk[0][0]])
            pretestb2.extend([Xk[0][1]])
            pretestc.extend(yhat[0])
        testa.extend([pretesta])
        testb.extend([pretestb])
        testa2.extend([pretesta2])
        testb2.extend([pretestb2])
        testc.extend([pretestc])
    plt.imshow(testa, cmap='gray', interpolation='nearest')   #plotear la segmentación de la primera neurona(activacion)
    plt.show()
    plt.imshow(testb, cmap='gray', interpolation='nearest')   #plotear la segmentación de la segunda neurona(activacion)
    plt.show()
    plt.imshow(testa2, cmap='gray', interpolation='nearest')   #plotear la segmentación de la primera neurona de la segunda capa(activacion)
    plt.show()
    plt.imshow(testb2, cmap='gray', interpolation='nearest')   #plotear la segmentación de la segunda neurona de la segunda capa(activacion)
    plt.show()
    plt.imshow(testc, cmap='gray', interpolation='nearest')   #plotear la segmentacion en la salida(activación)
    plt.show()
    plt.plot(error)
    plt.ylabel('some numbers')
    plt.show()
