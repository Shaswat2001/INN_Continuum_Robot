import argparse
from platform import node
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from Algorithms.Cascade_Correlation import CasCor
from Algorithms.ELM import ELM
from Algorithms.INN import INN
from Algorithms.MLP import MLP_Model
from Algorithms.RBFNet import RBFModel
from utils import *
from sklearn.metrics import r2_score

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)                 

def build_parse():

    parser = argparse.ArgumentParser(description="RL Algorithm Variables")

    parser.add_argument("robotModel",nargs="?",type=str,default="Static",help="Type of Robot Model for Kinematics")
    parser.add_argument("learningModel",nargs="?",type= str,default="ELM",help="Type of Learning Methodology to be used")
    parser.add_argument("normalised",nargs="?",type=bool,default=True,help="Normalize the dataset for training")
    parser.add_argument("isRotation",nargs="?",type=bool,default=False,help="Whether rotation is included in the learning process")
    parser.add_argument("scaleMethod",nargs="?",type=str,default="Standard",help="Type of Scaler Used")

    args = parser.parse_args()

    return args    
        
def train(T_train,W_train,T_test,W_test,model,args,giveMagnitude=False):
    
    model.fit(W_train,T_train)
    Ytr_pred = model.predict(W_train)
    Yts_pred = model.predict(W_test)

    f = open("Results/Training/"+args.learningModel+".sav","wb")
    pickle.dump(model,f)
    f.close()


    error = rmse(T_test,Yts_pred)
    error_tr = rmse(T_train,Ytr_pred)

    r2_test = r2_score(T_test,Yts_pred)
    
    if giveMagnitude:
        magnitude = errorMagnitude(T_test,Yts_pred)
        return (error_tr,error,r2_test,magnitude)
        
    return (error_tr,error,r2_test)

if __name__=="__main__":

    args = build_parse()
    T_train,T_test,W_train,W_test = load_dataset(args)

    if args.learningModel == "INN":

        INN(W_train,T_train)

    else:

        if args.learningModel == "KNN":
            model = KNeighborsRegressor(n_neighbors=40,weights='uniform')

        elif args.learningModel == "Decision":
            model = DecisionTreeRegressor(criterion='mae',splitter='best',min_samples_split=5,min_samples_leaf=3)

        elif args.learningModel == "Cascade":
            model = CasCor(W_train.shape[1],T_train.shape[1])

        elif args.learningModel == "ELM":
            model = ELM(nodes = 100)

        elif args.learningModel == "MLP":
            model = MLP_Model(W_train,T_train)

        elif args.learningModel == "RBF":
            model = RBFModel(W_train,T_train)

        val = train(T_train,W_train,T_test,W_test,model,args)