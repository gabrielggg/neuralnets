import numpy as np
import math
import matplotlib.pyplot as plt
#import pylab as plt

class NeuralNetwork:
    def __init__(self):
        #np.random.seed(10) # for generating the same results
       # self.wij   = np.random.rand(6,100) # input to hidden layer weights
        self.wij   =  np.random.uniform(low=-1, high=1, size=(6,100))
       # self.bj    = np.random.rand(100,1) # bias input
        self.bj   =  np.random.uniform(low=-1, high=1, size=(100,1))
       # self.wjk   = np.random.rand(100,27) # hidden layer to output weights
        self.wjk   =  np.random.uniform(low=-1, high=1, size=(100,27))
       # self.bk    = np.random.rand(27,1) # bias output
        self.bk   =  np.random.uniform(low=-1, high=1, size=(27,1))
        
    def sigmoid(self, x, w, b):
        z = np.dot(x, w) + b.T
        return 1/(1 + np.exp(-z))
    
    
    
    def sigmoid_derivative(self, x, w, b):
        return self.sigmoid(x, w, b) * (1 - self.sigmoid(x, w, b))

    def tanh(self, xx, w, b):
        #print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        #print(w)
        #print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        xx = np.dot(xx, w) 
        #print(x)
        t=(np.exp(xx)-np.exp(-xx))/(np.exp(xx)+np.exp(-xx))

        return t

    def tanh_derivative(self, x, w, b):
        x = np.dot(x, w) 
        t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        dt=1-t**2
        return dt

    def stable_softmax(self, x, w, b):
        X = np.dot(x, w) 
        #print(X)
        #print(np.shape(X))
        #print(np.max(X))
        exps = np.exp(X - np.max(X, 1, keepdims = True))
        #print(np.shape(exps.sum(1, keepdims = True)))
        #print((exps / exps.sum(1, keepdims = True)))
        #print((exps / exps.sum(1, keepdims = True)).sum(1, keepdims = True))
        return exps / exps.sum(1, keepdims = True)


    def softmax(self, x, w, b):
        X = np.dot(x, w) 
        exps = np.exp(X)
        return exps / exps.sum(1, keepdims = True)

    def delta_cross_entropy(self, yhaty,yy):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
            Note that y is not one-hot encoded vector. 
            It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        #print(yhaty[0],"zzzzzzzzzzzz")
        yhat_tmp = np.copy(yhaty)
        y_tmp = np.copy(yy)
        y_tmp = y_tmp.argmax(axis=1)
        m = y_tmp.shape[0]
        grad = yhat_tmp
        grad[range(m),y_tmp] -= 1
        grad = grad/m
        #print(yhaty[0],"ppppppppppppppppp")
        #print(grad,"gggggggggggggggggggg")
        return grad

    def cross_entropy(self,yhaty,yy):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
            Note that y is not one-hot encoded vector. 
            It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        yhat_tmp = np.copy(yhaty)
        y_tmp = np.copy(yy)
        y_tmp = y_tmp.argmax(axis=1)
        m = y_tmp.shape[0]
        p = yhat_tmp
        # We use multidimensional array indexing to extract 
        # softmax probability of the correct label for each sample.
        # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
        #print(p[range(m),y])
        log_likelihood = -np.log(p[range(m),y_tmp])
        loss = np.sum(log_likelihood) / m
        return loss

    
    def gradient_descent(self, x, y, iterations):
        for i in range(iterations):
            #batch gradient descent
            #batch gradient descent
            #if (i  % 10000 == 0):
            #sampleamos nuestro dataset
            #if (i  % 20 == 0):
                #idx = np.random.choice(np.arange(len(x)), 10000, replace=False)
                #x = x[idx]
                #y = y[idx]
            idx = np.random.choice(np.arange(len(x)), 100, replace=False)
            x = x[idx]
            y = y[idx]
            Xi = x
            Xj = self.tanh(Xi, self.wij, self.bj)
            #print(Xj)
            #print(Xj)
            #print(y)
            yhat = self.stable_softmax(Xj, self.wjk, self.bk)
            #print(yhat[0])
            yhat = np.clip(yhat, 1e-7, 1 - 1e-7)
            #yhat_tmp = np.copy(yhat)
            #y_tmp = np.copy(y)
            #yhat_tmp2 = np.copy(yhat)
            #y_tmp2 = np.copy(y)
            #print(yhat[0], "yyyyyyyyyyyyyyyyyyyyyyyy")
            #print(yhat.sum(1, keepdims = True))
            
            #yhat = np.exp(yhat)
            g_wjk = np.dot( Xj.T, self.delta_cross_entropy(yhat,y))
            #print(yhat[0], "xxxxxxxxxxxxxxxxxxxxxxxx")
            #print(g_wjk, np.shape(g_wjk))
            #g_wij = 
            

            g_wij = np.dot(Xi.T, np.dot( self.delta_cross_entropy(yhat,y), self.wjk.T) * self.tanh_derivative(Xi, self.wij, self.bj))
            #yhat = yhat / yhat.sum(1, keepdims = True)
           #print(yhat[0],"zzzz")
            #print(sum(yhat[0]))
            #print(self.wij)
            #print(np.shape(y))
            #print(np.shape(yhat))
            #print(sum(-np.log(yhat))/len(yhat))
            np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
            #print(sum((y[0] - yhat[0]).T), "test")
            #print(yhat[0], sum(yhat[0]))
            #print(y[0], sum(y[0]))
            #print(yhat[0])
            #print(y[0] - yhat[0])
            #print(sum(y[0] - yhat[0]))

            #print(yhat[0])
            #print(y[0])
            #print(y[0]-yhat[0])
            #print(sum((y[0]-yhat[0]).T))
            #yhat[0][0]=5.3
            
            #print(yhat[0])
            #print(sum(sum(y-yhat)),"yyyyyyy")
            # gradients for hidden to output weights
            #print(Xj)
            #print(y.argmax(axis=1))
            
            #g_wjk = np.dot(Xj.T, (-1+(1/(yhat[y.argmax(axis=1)]))))

            #print(np.shape(g_wjk))
            #print(g_wjk,"wwww")
            # gradients for input to hidden weights
            #los gradientes(derivadas) son la razón de cambio del error con respecto a los pesos por lo tanto si la razón de cambio es positiva significa
            #que la pendiente de la recta es positiva por lo tanto el error va en aumento y hay que restar el gradiente de los pesos.
            #print((-1+(y/(yhat))))
            #g_wij = np.dot(Xi.T, np.dot((-1+(y/(yhat))), self.wjk.T) * self.tanh_derivative(Xi, self.wij, self.bj))
            # gradients for input to hidden bias
            #print(g_wij)
            #print(g_wij,"wwweeeew")
            #g_bk = np.sum(np.dot(1, (y - yhat) * self.sigmoid_derivative(Xj, self.wjk, self.bk)).T, axis=1, keepdims=True)
            #g_bk = np.sum(((y-yhat) * self.sigmoid_derivative(Xj, self.wjk, self.bk)).T, axis=1, keepdims=True)
            #g_bj = np.sum(np.dot(1, np.dot((y - yhat) * self.sigmoid_derivative(Xj, self.wjk, self.bk), self.wjk.T) * self.sigmoid_derivative(Xi, self.wij, self.bj)).T, axis=1, keepdims=True)
            #g_bj = np.sum((np.dot((y-yhat) * self.sigmoid_derivative(Xj, self.wjk, self.bk), self.wjk.T) * self.tanh_derivative(Xi, self.wij, self.bj)).T, axis=1, keepdims=True)
            # update weights and bias we sum the gradients because the MSE(error cuadrático medio ) is calculated like this (yhat-y) and we are using (y-yhat) so there is the change of sign
            self.wij -= g_wij * 10
            self.wjk -= g_wjk * 10
            #self.bj  += g_bj  * 0.1
            #self.bk  += g_bk  * 0.1
            #error.extend(sum((y - yhat).T))
            #yhat_tmp3 = np.copy(yhat)
            #y_tmp3 = np.copy(y)
            loss = self.cross_entropy(yhat,y)
            #print(loss)
            error.append(loss)
            
        #print('The final prediction from neural network are: ')
        #print(yhat)
        self.yhat = yhat
        #print(self.yhat[0], "xxxxxxxxxxxxxxxxxxxxxxxx")
        self.y = y
        #print(np.shape(yhat))
        #error = np.reshape(error, (1, 100000))
        #print(np.shape(error))


if __name__ == '__main__':
    neural_network = NeuralNetwork()
    #print('Random starting input to hidden weights: ')
    #print(neural_network.wij)
    #print('Random starting hidden to output weights: ')
    #print(neural_network.wjk)
    
    # read in all the words
    words = open('names.txt', 'r').read().splitlines()
    #words = words[:8]

    # build the vocabulary of characters and mappings to/from integers
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}
    #print(itos)

    # build the dataset

    block_size = 3 # context length: how many characters do we take to predict the next one?
    X, Y = [], []
    for w in words:
    
    #print(w)
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            #print(''.join(itos[i] for i in context), '--->', itos[ix])
            context = context[1:] + [ix] # crop and append
    
    #print(X, np.shape(X))
    #print(Y, np.shape(Y))
    n_values = 26 + 1
    #print(np.eye(n_values)[Y])
    Y = np.eye(n_values)[Y]
    #print(Y)
    

    C = np.random.rand(27,2)
    C = np.random.uniform(low=-1, high=1, size=(27,2))
    #print(C)
    
    #print(init)
    
    #print(np.shape(X))
    emb = C[X]
    #print(emb)
    #print(emb[0])
    #print(emb)
    #print(len(X), "leeeen")
    #print(emb[0],emb[1])
    #print(X[13])
    #print(C[13])
    #print(emb)
    #print(np.shape(emb))
    emb = np.reshape(emb, (len(X), 6))
    #print(emb)


    #X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    #y = np.array([[0, 1, 1, 0]]).T
    X = emb
    y = Y
    error = []
    testa = []
    testb = []
    testc = []
    pretesta = []
    pretestb = []
    pretestc = []
    neural_network.gradient_descent(X, y, 2500)
    #print(neural_network.y)
    #sampling
    
    #print(init)
    #print(C)
    #init[0][4:] = C[4]
    #print(init)
    #print(init[0][4:])
    
    letters = [".", "a", "b","c", "d", "e","f", "g", "h","i", "j", "k","l", "m", "n","o", "p", "q","r", "s", "t","u","v","w", "x", "y","z"]
    for u in range (0,10):
        name = ""
        init = C[0]
        #print(init)
        init = np.resize(init, (1, 6))
        for i in range (0,25):
            

        
        
            xx = neural_network.tanh(init, neural_network.wij, neural_network.bj)
                #print(Xj)
                #print(Xj)
                #print(y)
            predic = neural_network.stable_softmax(xx, neural_network.wjk, neural_network.bk)
            
            sample = np.random.choice(np.arange(0, 27, 1), 1, p=predic[0])
            if i==0:
                sample= np.random.randint(1,26)
                #print(sample)
                init[0][0:2] = init[0][2:4] 
                init[0][2:4] = init[0][4:] 
                init[0][4:] = C[sample]
                #print(init)
                name = name+letters[sample]
            elif sample[0] == 0:
                #print(name)
                break
            else:
                init[0][0:2] = init[0][2:4] 
                init[0][2:4] = init[0][4:] 
                init[0][4:] = C[sample[0]]
                #print(init)
                #print(sample)
                name = name+letters[sample[0]]
        print(name)
        

    #print(np.shape(neural_network.yhat))
    #print(sum(neural_network.yhat[0]))
    #print(np.arange(32))
    #print(predic)
    #print(np.arange(0, 27, 1))
    #sample = np.random.choice(np.arange(0, 27, 1), 1, p=predic[0])
    #print(sample)
    #print(np.shape(neural_network.y))
    #print(neural_network.y[0].astype(int))
    ind = []
    for c in range (0,100):
       #ind[c] = neural_network.y[c].astype(int).argmax(axis=0)
     #  print(neural_network.y[c].astype(int).argmax(axis=0))
       ind.append(neural_network.yhat[c][neural_network.y[c].argmax(axis=0)])
       #ind.append(neural_network.yhat[0])
       #print(neural_network.yhat[c])
       #print(neural_network.y[c].astype(int).argmax(axis=0)) 
    #print(neural_network.y[999].astype(int).argmax(axis=0))
    #print(neural_network.y[999].astype(int))
    #print(ind)
    #print()
    #print(neural_network.yhat[np.arange(32), neural_network.y.astype(int)])
    #print('Final input to hidden weights: ')
    #print(neural_network.wij)
    #print('Final hidden to output weights: ')
    #print(neural_network.wjk)
    #print('Final input to hidden bias: ')
    #print(neural_network.bj)
    #print('Final hidden to output bias: ')
    #print(neural_network.bk)
    #Xi = init
    #Xj = neural_network.tanh(Xi, neural_network.wij, neural_network.bj)
            #print(Xj)}
            #print(y)
    
    #yhat = neural_network.sigmoid(Xj, neural_network.wjk, neural_network.bk)
    #print(yhat)
    # for i in range(0,100):
    #     pretesta = []
    #     pretestb = []
    #     pretestc = []
    #     for j in range(0,100):
    #         x1 = i/100
    #         x2 = j/100
    #         #Xi = [x1, x2]
    #         Xi = np.array([x1, x2])
    #         Xj = neural_network.sigmoid(Xi, neural_network.wij, neural_network.bj)
    #         #print(Xj[0])
    #         #print(Xj)
    #         #print(Xj[0][1])
    #         yhat = neural_network.sigmoid(Xj, neural_network.wjk, neural_network.bk)
    #         pretesta.extend([Xj[0][0]])
    #         pretestb.extend([Xj[0][1]])
    #         pretestc.extend(yhat[0])
    #     testa.extend([pretesta])
    #     testb.extend([pretestb])
    #     testc.extend([pretestc])
    #plt.imshow(testa, cmap='gray', interpolation='nearest')   #plotear la segmentación de la primera neurona(activacion)
    #plt.show()
    #plt.imshow(testb, cmap='gray', interpolation='nearest')   #plotear la segmentación de la segunda neurona(activacion)
    #plt.show()
    #plt.imshow(testc, cmap='gray', interpolation='nearest')   #plotear la segmentacion en la salida(activación)
    #plt.show()
    plt.plot(error)
    plt.ylabel('loss')
    plt.show()
