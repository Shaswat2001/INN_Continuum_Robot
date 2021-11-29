import numpy as np

def sigmoid(Z):
    
    A = 1/(1+np.exp(-Z))
    return A

def tanh(Z):
    
    A = np.tanh(Z)
    return A

def sigmoid_backward(Z):
    
    A = 1/(1+np.exp(-Z))
    deriv = A*(1-A)
    return deriv

def tanh_backward():
    pass