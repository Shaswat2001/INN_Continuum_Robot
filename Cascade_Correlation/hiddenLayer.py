import numpy as np
from utils import sigmoid,sigmoid_backward
import math
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
        self.max_iterations = 50

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

        