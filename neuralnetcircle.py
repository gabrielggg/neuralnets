import numpy as np
import matplotlib.pyplot as plt
#import pylab as plt

def isInside(circle_x, circle_y, rad, x, y):
     
        # Compare radius of circle
        # with distance of its center
        # from given point
        if ((x - circle_x) * (x - circle_x) +
            (y - circle_y) * (y - circle_y) <= rad * rad):
            return True
        else:
            return False

class NeuralNetwork:
    def __init__(self):
        #np.random.seed(10) # for generating the same results
        #para poder generalizar un dataset de punto dentro de un círculo necesitamos por lo menos 3 neuronas y que con 2 solo podemos
        #generar 2 líneas en cambio con 3 ya podemos generar 3 líneas y un triangulo con forma de círculo gracias a las sigmoides
        #  en la segmentación
        self.wij   = np.random.rand(2,3) # input to hidden layer weights
        self.bj    = np.random.rand(3,1) # bias input
        self.wjk   = np.random.rand(3,1) # hidden layer to output weights
        self.bk    = np.random.rand(1,1) # bias output
        
    def sigmoid(self, x, w, b):
        z = np.dot(x, w) + b.T
        return 1/(1 + np.exp(-z))
    
    def sigmoid_derivative(self, x, w, b):
        return self.sigmoid(x, w, b) * (1 - self.sigmoid(x, w, b))
    
    
    
    def gradient_descent(self, x, y, iterations):
        for i in range(iterations):
            #batch gradient descent
            #batch gradient descent
            #if (i  % 10000 == 0):
            #sampleamos nuestro dataset
            if (i  == 0):
                idx = np.random.choice(np.arange(len(x)), 1000, replace=False)
                x = x[idx]
                y = y[idx]
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
            self.wij += g_wij * 0.01
            self.wjk += g_wjk * 0.01
            self.bj  += g_bj * 0.01
            self.bk  += g_bk * 0.01          
            error.extend([y[0]-yhat[0]])
        print('The final prediction from neural network are: ')
        print(yhat)

if __name__ == '__main__':
    neural_network = NeuralNetwork()
    print('Random starting input to hidden weights: ')
    print(neural_network.wij)
    print('Random starting hidden to output weights: ')
    print(neural_network.wjk)
    #generando dataset
    circle_x = 0.5
    circle_y = 0.5
    rad = 0.3
    arrinput = []
    arroutput = []
    unosx = []
    unosy = []
    zeros = []
    for ii in range(0,50):
        for jj in range(0,50):
            xx = (ii*2)/100
            yy = (jj*2)/100
            arrinput.append([xx,yy])
            if(isInside(circle_x, circle_y, rad, xx, yy)):
                unosx.append(xx)
                unosy.append(yy)
                arroutput.append(1)
            else:
                #print("Outside")
                zeros.append([xx,yy])
                arroutput.append(0)
    #indices1 = [i for i, x in enumerate(arroutput) if x == 1]
    #indices0 = [i for i, x in enumerate(arroutput) if x == 0]
    unosx = np.array(unosx)
    unosy = np.array(unosy)
    print(unosx)
    plt.scatter(unosx,unosy)
    plt.show()
    #print(len(indices1), len(indices0))
    #X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    #y = np.array([[0, 1, 1, 0]]).T
    X = np.array(arrinput, float)
    y = np.array([arroutput], int).T
    
    print(len(X))
    print(len(y))
    error = []
    testa = []
    testb = []
    testc = []
    testd = []
    pretesta = []
    pretestb = []
    pretestc = []
    pretestd = []
    neural_network.gradient_descent(X, y, 10000)
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
        pretestd = []
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
            #solo vamos a imprimir las activaciones de las primeras 3 neuronas y de la salida
            pretesta.extend([Xj[0][0]])
            pretestb.extend([Xj[0][1]])
            pretestc.extend([Xj[0][2]])
            pretestd.extend(yhat[0])
        testa.extend([pretesta])
        testb.extend([pretestb])
        testc.extend([pretestc])
        testd.extend([pretestd])
    #print(round(testd))
    rounded = [x for x in testd]
    rounded = [np.round(x) for x in rounded]
    Xi = np.array([0, 0])
    Xj = neural_network.sigmoid(Xi, neural_network.wij, neural_network.bj)
            #print(Xj[0])
            #print(Xj)
            #print(Xj[0][1])
    yhat = neural_network.sigmoid(Xj, neural_network.wjk, neural_network.bk)
    print(yhat,'yhat')
    Xi = np.array([0.5, 0.5])
    Xj = neural_network.sigmoid(Xi, neural_network.wij, neural_network.bj)
            #print(Xj[0])
            #print(Xj)
            #print(Xj[0][1])
    yhat = neural_network.sigmoid(Xj, neural_network.wjk, neural_network.bk)
    print(yhat,'yhat')
    #print(rounded)
    plt.imshow(testa, cmap='gray', interpolation='nearest')   #plotear la segmentación de la primera neurona(activacion)
    plt.show()
    plt.imshow(testb, cmap='gray', interpolation='nearest')   #plotear la segmentación de la segunda neurona(activacion)
    plt.show()
    plt.imshow(testc, cmap='gray', interpolation='nearest')   #plotear la segmentacion de la tercera neurona(activación)
    plt.show()
    plt.imshow(testd, cmap='gray', interpolation='nearest')   #plotear la segmentacion en la salida(activación)
    plt.show()
    plt.plot(error)
    plt.ylabel('error')
    plt.show()
