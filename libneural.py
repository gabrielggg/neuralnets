import numpy as np
import matplotlib.pyplot as plt
#import pylab as plt



class NeuralNetwork:
    def __init__(self, hidden_layers, neural_net_architecture, factor):
        #np.random.seed(10) # for generating the same resultss
        self.wv = []
        self.bv = []
        self.error = []
        self.hidden_layers = hidden_layers
        self.neural_net_hidden_architecture = neural_net_architecture[1:-1]
        self.total_neurons = sum(self.neural_net_hidden_architecture)
        self.neuron_activation_images_arrays = []
        for _ in range (self.total_neurons):
            self.neuron_activation_images_arrays.append([])
        print(self.neuron_activation_images_arrays)
        self.factor = factor
        total_layers = hidden_layers +2 #sumamos la capa de neuronas de entrada y la capa de neuronas de salida
        #generate weights and biases for all the layers
        for i in range(total_layers-1):
            self.wv.append(np.random.rand(neural_net_architecture[i],neural_net_architecture[i+1]))
            self.bv.append(np.random.rand(neural_net_architecture[i+1],1))
        
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
            #batch gradient descent
            #batch gradient descent
            #if (i  % 10000 == 0):
            #sampleamos nuestro dataset
            if (i  == 0):
                idx = np.random.choice(np.arange(len(x)), 1000, replace=False)
                x = x[idx]
                x_real = x
                y = y[idx]
            for z in range (len(self.wv)):
                xv.append(x)
                x = self.sigmoid(xv[z], self.wv[z], self.bv[z])
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
                    gwv = gwv[::-1]
                    gbv = gbv[::-1]
                    for zx in range (len(self.wv)): 
                        self.wv[zx] += gwv[zx] * self.factor
                        self.bv[zx] += gbv[zx] * self.factor    
                    self.error.extend([y[0]-yhat[0]])
        
        print('The final prediction from neural network are: ')
        print(yhat)
        print(gwv)

    def test(self):
        testd = []        
        for i in range(0,100):
                pretest = []
                pretestd = []
                values_arr_x = []
                for j in range(0,100):
                    x1 = i/100
                    x2 = j/100
                    Xi = np.array([x1, x2])
                    values_arr_x = []
                    values_arr_x.append(Xi)
                    for u in range (self.hidden_layers): 
                        values_arr_x.append(self.sigmoid(values_arr_x[u], self.wv[u], self.bv[u]))
                        first, last = np.shape(values_arr_x[u+1])
                        pretest.append([])
                        for neuron_index in range (last):
                            pretest[u].append([])
                            pretest[u][neuron_index].extend([values_arr_x[u+1][0][neuron_index]])
                    yhat = self.sigmoid(values_arr_x[u+1], self.wv[u+1], self.bv[u+1])
                    pretestd.extend(yhat[0])
                counter_neurons=0
                for idx in enumerate(self.neural_net_hidden_architecture):
                    for neuronx in range(idx[1]):
                        self.neuron_activation_images_arrays[counter_neurons].extend([pretest[idx[0]][neuronx]])
                        counter_neurons = counter_neurons +1
                testd.extend([pretestd])
        counter_neurons=0
        fig, ax = plt.subplots(nrows=max(self.neural_net_hidden_architecture), ncols=len(self.neural_net_hidden_architecture)+1, figsize=(10,7))
        for idx in enumerate(self.neural_net_hidden_architecture):
            for neuronx in range(idx[1]):                
                ax[neuronx][idx[0]].imshow(self.neuron_activation_images_arrays[counter_neurons], cmap='gray', interpolation='nearest')
                counter_neurons = counter_neurons +1
        ax[0][idx[0]+1].imshow(testd, cmap='viridis_r', interpolation='nearest')
        plt.tight_layout()
        plt.show()
        plt.plot(self.error)
        plt.ylabel('some numbers')
        plt.show(block=True)
        
        





