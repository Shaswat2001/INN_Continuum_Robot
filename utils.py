import numpy as np
import math
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

def load_dataset(args):

    f = open("Dataset/" + args.robotModel + ".pkl","rb")
    data = pickle.load(f)
    f.close()

    if args.isRotation:
        W = data[:,4:]
    else:
        W = data[:,4:7]
        
    if args.robotModel == "Static":
        minTension = 5
        maxTension = 15

    else:

        minTension = 310
        maxTension = 340

    T = data[:,:4]
    T = normalize_tension(T,minTension,maxTension)

    T_train,T_test,W_train,W_test = train_test_split(T,W,test_size=0.15,random_state=0)
    
    return T_train,T_test,W_train,W_test

def cost(y_test,y_pred):
    '''
    Calculates error of the model
    '''
    error = (y_test-y_pred)/y_test
    error = np.sum(abs(error))/(y_test.shape[0]*y_test.shape[1])*100
    
    return error

def rmse(y_test,y_pred):
    error = np.sum((y_test-y_pred)**2)
    error = error/(y_test.shape[0]*y_test.shape[1])
    error = math.sqrt(error)
    return error

def errorMagnitude(y_true,y_pred):
    
    minMag = min([min(abs(i)) for i in y_true-y_pred])
    maxMag = max([max(abs(i)) for i in y_true-y_pred])
    
    return (minMag,maxMag)

def normalize_tension(tension,minTension,maxTension):
    z = (tension - minTension)/(maxTension-minTension)
    return z

def denormalize_tension(tension,minTension,maxTension):
    z = tension*(maxTension-minTension) + minTension
    return z

