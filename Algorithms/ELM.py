import numpy as np
import pandas as pd
import math

class ELM():
    
    def __init__(self,nodes):
        self.neurons = nodes
        self.lmb = 56
    
    def sigmoid(self,z):
        
        s = 1/(1+np.exp(-z))
        return s

    def relu(self,z):
        s = np.maximum(0,z)
        return s

    def tanh(self,z):
        s = np.tanh(z)
        return s
    
    def initlialization(self,X):
        value = 1/math.sqrt(X.shape[1])
        self.W = np.random.uniform(-value,value,(self.neurons,X.shape[1]))
        self.b = np.ones(shape=(self.neurons,1))
        
    def fit(self,X,Y):
        
        self.initlialization(X)
        z = np.dot(self.W,X.T)+self.b
        H = self.relu(z)
        self.B = np.dot(Y.T,np.linalg.pinv(H))
        # Hm = np.linalg.inv(np.dot(H,H.T)+1/self.lmb)
        # Hm = np.dot(H.T,Hm)
        # self.B = np.dot(Y,Hm)
    
    def predict(self,X):
        z = np.dot(self.W,X.T)+self.b
        H = self.relu(z)
        y_pred = np.dot(self.B,H)
        
        return y_pred.T