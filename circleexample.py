import libneural
import numpy as np
import matplotlib.pyplot as plt

def isInside(circle_x, circle_y, rad, x, y):
        if ((x - circle_x) * (x - circle_x) +
            (y - circle_y) * (y - circle_y) <= rad * rad):
            return True
        else:
            return False

if __name__ == '__main__':
    neural_network = libneural.NeuralNetwork(3, [2,3,5,3,1], 0.01)
    print('Random starting input to hidden weights: ')
    print(neural_network.wv[0])
    print('Random starting hidden to output weights: ')
    print(neural_network.wv[1])
    
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
    unosx = np.array(unosx)
    unosy = np.array(unosy)
    print(unosx)
    plt.scatter(unosx,unosy)
    plt.show()
    X = np.array(arrinput, float)
    y = np.array([arroutput], int).T
    print(len(X), "longitud x")
    print(X)
    print(len(y), "length y")
    print(y)
    neural_network.gradient_descent(X, y, 50000)
    print('Final input to hidden weights: ')
    print(neural_network.wv[0])
    print('Final hidden to output weights: ')
    print(neural_network.wv[1])
    print('Final input to hidden bias: ')
    print(neural_network.bv[0])
    print('Final hidden to output bias: ')
    print(neural_network.bv[1])
    neural_network.test()