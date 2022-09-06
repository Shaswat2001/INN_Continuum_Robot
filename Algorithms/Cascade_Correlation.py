import numpy as np
import math
import pandas as pd

def sigmoid(Z):
    
    A = 1/(1+np.exp(-Z))
    return A

def sigmoid_backward(Z):
    
    A = 1/(1+np.exp(-Z))
    deriv = A*(1-A)
    return deriv

class HiddenLayer():

    def __init__(self,input_size,output_size,candidates):
        self.I = input_size
        self.O = output_size
        self.K = candidates
        self.tp = 'xavier'
        self.converged = False
        self.learning_rate = 0.01
        self.weights = self.initialize()
        self.patience = 0.01
        self.max_iterations = 100

    def initialize(self):

        if self.tp == 'xavier':
            value = 1/math.sqrt(self.I)
            weight = np.random.uniform(-value,value,(self.K,self.I))
        else:
            weight = np.random.randn(self.K,self.I)
        return weight
    
    def forward(self,X,index=None):

        if index==None:
            Z = np.dot(X,self.weights.T)
            A = sigmoid(Z)
        else:
            Z = np.dot(X,self.weights[index,:].T)
            A = sigmoid(Z)

        return Z,A
    
    def getCorrelation(self,V,E):

        v_term = V - np.mean(V,axis=0)
        e_term = E - np.mean(E,axis=0)
        corr = np.dot(v_term.T,e_term)
        ss = np.sum(np.abs(corr), axis=1)

        return ss,e_term,corr
    
    def backward_prop(self,xs,vs,corr,e_term):
        
        dweights = np.zeros(self.weights.shape)
        
        for k in range(self.K):
            tmp1 = np.multiply(sigmoid_backward(vs[:, k])[:, np.newaxis], xs).T
            tmp2 = np.dot(tmp1, e_term)
            tmp3 = np.dot(np.sign(corr[k, :]), tmp2.T)
            dweights[k, :] = tmp3
        
        return dweights

    def update_weights(self,gradient):
        self.weights = self.weights-self.learning_rate*gradient
    
    def check_convergence(self,iteration):
        if self.ss.shape[0] < 3:
            self.converged = False
        elif iteration == self.max_iterations:
            self.converged = True
        
        else:
            diff = np.abs(self.ss[-1,:] - self.ss[-2,:])
            if np.mean(diff) < self.patience:
                self.converged = True
            else:
                self.converged = False
    
    def add_ss(self,ss):
        if not hasattr(self, 'ss'):
            self.ss = np.array([ss])
            
        else:
            new_ss = np.zeros((self.ss.shape[0] + 1, self.ss.shape[1]))
            new_ss[:-1, :] = self.ss
            new_ss[-1, :] = ss
            self.ss = new_ss

    
    def train(self,X,losses):

        itr = 0
        while not self.converged:
            
            hs,vs = self.forward(X)
            ss,e_term,corr = self.getCorrelation(vs,losses)
            grads = self.backward_prop(X,vs,corr,e_term)
            self.update_weights(grads)
            self.add_ss(ss)
            self.check_convergence(itr)

            itr+=1
        
        ss, _, _ = self.getCorrelation(vs, losses)
        self.best_candidate_idx = np.argmax(ss)

    def get_best_candidate_values(self, xs):
        hs, vs = self.forward(xs, index=self.best_candidate_idx)
        return vs

class CasCor():

    def __init__(self,input_size,output_size):
        self.I = input_size
        self.O = output_size
        self.tp = 'xavier'
        self.weights = self.initialize()
        

        self.max_iteration = 100    
        self.max_iteration_io = 500
        self.train_loss = []
        self.test_loss = []
        self.hidden_units = []
        self.training_accuracy=[]
        self.testing_accuracy=[]
        self.acceptable_loss = 0.001
        
        # Training hyperParameters
        self.learning_rate = 0.04
        self.activation = 'sigmoid'
        self.patience = 0.001
        self.miniBatch_size = 64

    def initialize(self):

        if self.tp == 'xavier':
            value = 1/math.sqrt(self.I)
            weight = np.random.uniform(-value,value,(self.O,self.I))
        else:
            weight = np.random.randn(self.O,self.I)
        return weight

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
            loss = np.sqrt((y_true-y_pred)**2/(y_true.shape[0]*y_true.shape[1]))
        else:
            loss = np.sqrt(np.sum((y_true-y_pred)**2/(y_true.shape[0]*y_true.shape[1])))
        
        return loss
    
    def get_magnitude(self,y_true,y_pred):
        minMag = min([min(abs(i)) for i in y_true-y_pred])
        maxMag = max([max(abs(i)) for i in y_true-y_pred])

        return (minMag,maxMag)
    
    def backward_prop(self,X,y_true,y_pred):
        grad = np.zeros(self.weights.shape)

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
        elif len(self.train_loss)>=2 and abs(self.train_loss[-1]-self.train_loss[-2])<self.patience:
            self.converged = True
        
    def augment_input(self,xs,vs):
        new_xs = np.zeros((xs.shape[0], xs.shape[1] + 1))
        new_xs[:, :-1] = xs
        new_xs[:, -1] = vs
        
        return new_xs
    
    def addHiddenLayer(self,X,Y,losses):

        candidates_pool = HiddenLayer(self.I, self.O, 5)
        candidates_pool.train(X, losses)		
        vs = candidates_pool.get_best_candidate_values(X)
        xs = self.augment_input(X, vs)
        
        self.hidden_units.append(candidates_pool)
        self.I += 1 	# just added one more element for each input, so the size fo the input has increased

        new_weights = self.initialize()
        new_weights[:, :-1] = self.weights
        self.weights = new_weights
        
        return xs

    def calculate_accuracy(self,y_true,y_pred):
        error = (y_true-y_pred)/y_true
        error = np.sum(abs(error))/(y_true.shape[0]*y_true.shape[1])*100
        return error

    def evaluateNetwork(self,X,Y,X_test,Y_test):

        _,vs = self.forward(X)
        error_train = self.calculate_accuracy(Y,vs)
        loss = self.cost_calculation(Y,vs)
        self.train_loss.append(loss)
        self.training_accuracy.append(error_train)
        
        _,vs = self.forward(X_test)
        error_test = self.calculate_accuracy(Y_test,vs)
        loss = self.cost_calculation(Y_test,vs)
        self.test_loss.append(loss)
        self.testing_accuracy.append(error_test)


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
            
            self.train_loss.append(total_loss/len(minibatch))
            self.evaluateNetwork(X,Y,X_test,Y_test)
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
            itr+=1
            X = self.addHiddenLayer(X,Y,losses)
            

    




