import numpy as np 
import pandas as pd
from utils import sigmoid,sigmoid_backward
from hiddenLayer import HiddenLayer

class CasCor():

    def __init__(self,input_size,output_size):
        self.I = input_size
        self.O = output_size
        self.weights = self.initialize()

        self.max_iteration = 30
        self.max_iteration_io = 10
        self.train_loss = []
        self.test_loss = []
        self.acceptable_loss = 0.001
        
        # Training hyperParameters
        self.learning_rate = 0.01
        self.activation = 'sigmoid'
        self.patience = 0.001
        self.miniBatch_size = 64

    def initialize(self):

        weights = np.random.rand(self.O,self.I)
        return weights

    def add_Data(self,X_train,X_test,Y_train,Y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
    
    def minibatchData(self,X,Y):

        minibatches = []
        shuffled_index = np.random.permutation(X.shape[0])
        X_shuffled = X[shuffled_index]
        Y_shuffled = Y[shuffled_index]
        for i in range(X.shape[0]//self.miniBatch_size):
            X_mini = X_shuffled[i*self.miniBatch_size:(i+1)*self.miniBatch_size,:]
            Y_mini = Y_shuffled[i*self.miniBatch_size:(i+1)*self.miniBatch_size,:]
            minibatches.append((X_mini,Y_mini))

        if X.shape[0]%self.miniBatch_size!=0:
            X_mini = X_shuffled[(i+1)*self.miniBatch_size:,:]
            Y_mini = Y_shuffled[(i+1)*self.miniBatch_size:,:]
            minibatches.append((X_mini,Y_mini))
        
        return minibatches
    
    def forward(self,X):

        Z = np.dot(X,self.weights[:,:len(X[0])].T)

        if self.activation == 'sigmoid':
            A = sigmoid(Z)
        
        return Z,A
    
    def cost_calculation(self,y_true,y_pred,returnSum=True):

        if not returnSum:
            loss = 0.5*(y_true-y_pred)**2/y_true.shape[0]
        else:
            loss = np.sum(0.5*(y_true-y_pred)**2/y_true.shape[0])
        
        return loss
    
    def backward_prop(self,X,y_true,y_pred):
        grad = np.zeros(self.weights.shape[0])

        for i in range(y_true.shape[0]):
            delta = -(y_true[i]-y_pred[i])*sigmoid_backward(y_pred[i])
            grad+=np.outer(delta,X[i])
        
        grad/=self.miniBatch_size

        return grad
    
    def update_weights(self,grad):
        self.weights = self.weights - self.learning_rate*grad
    
    def check_convergence(self,iteration):
        
        if iteration==self.max_iteration_io:
            self.converged = True
        elif len(self.total_loss)>=2 and abs(self.total_loss[-1]-self.total_loss[-2])<self.patience:
            self.converged = True
    
    def addHiddenLayer(self):
        pass

    def evaluateNetwork(self):
        pass

    def plotLoss(self,loss):
        pass
    
    def train_io(self,X,Y,X_test,Y_test):
        
        itr = 0
        while not self.converged:
            minibatch = self.minibatchData(X,Y)
            total_loss = 0
            for (mini_X,mini_Y) in minibatch:
                
                hs,vs = self.forward(mini_X)
                loss = self.cost_calculation(vs,mini_Y)
                total_loss+=loss
                gradient = self.backward_prop(mini_X,mini_Y,vs)
                self.update_weights(gradient)
            
            self.evaluateNetwork(self,X_test,Y_test)
            self.train_loss.append(total_loss/len(minibatch))
            self.check_convergence(itr)
            itr+=1
            
    def train(self,X,Y,X_test,Y_test):

        itr = 0
        while True:
            self.converged = False
            self.train_io(X,Y,X_test,Y_test)

            _,vs = self.forward(X)
            losses = self.cost_calculation(Y,vs,returnSum=False)

            loss_sum = np.sum(losses)

            if loss_sum<=self.acceptable_loss:
                break

            if itr == self.max_iteration:
                break
                
            self.addHiddenLayer()
            

    




