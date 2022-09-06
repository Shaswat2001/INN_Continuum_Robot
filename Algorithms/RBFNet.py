from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import math

from keras.initializers import Initializer,Constant
from keras.engine.topology import Layer
import keras.backend as K
from tensorflow.keras.metrics import RootMeanSquaredError
from keras.layers import Dense,Conv1D,Conv2D,Flatten,MaxPool1D,LeakyReLU,MaxPooling1D,BatchNormalization,Input,LSTM,normalization,ReLU,Add
from keras.layers import Dense,Activation
from keras.models import Sequential
import tensorflow.keras

class KMeansCluster(Initializer):

    def __init__(self,X):
        self.X = X
    
    def __call__(self,shape,dtype=None,**kwargs):
        n_clusters = shape[0]
        k_means = KMeans(n_clusters=n_clusters)
        k_means.fit(self.X)

        return k_means.cluster_centers_
    
class RBFLayer(Layer):

    def __init__(self,output_dim,initializer,betas,**kwargs):
        self.output_dim = output_dim
        self.betas = betas
        self.initializer = initializer
        super(RBFLayer,self).__init__(**kwargs)
    
    def build(self,input_shape):

        self.centers = self.add_weight("centers",shape=(self.output_dim,input_shape[1]),initializer=self.initializer,trainable=True)
        self.betas = self.add_weight("betas",shape=(self.output_dim,1),initializer=Constant(value=self.betas),trainable=True)

        super(RBFLayer,self).build(input_shape)
    
    def call(self,X):
        CTR = K.expand_dims(self.centers)
        Distance = K.sum(K.square(K.transpose(X)-CTR),axis=1)
        return K.exp(K.transpose((-self.betas*Distance)))
    
    def compute_output_shape(self,input_shape):
        return(input_shape[0],self.output_dim)

def RBFModel(X_train,Y_train):

    input_layer = Input(shape=(X_train.shape[-1],))
    L1 = RBFLayer(output_dim=32,initializer=KMeansCluster(X_train),betas=0.1)(input_layer)
    L5 = Dense(units=64,activation='relu',kernel_initializer='uniform')(L1)
    #L6 = BatchNormalization()(L5)
    L6 = Dense(units=64,activation='relu',kernel_initializer='uniform')(L5)
    L7 = Dense(units=64,activation='relu',kernel_initializer='uniform')(L6)
    L8 = Add()([L5,L7])
    L9 = Dense(units=128,activation='relu',kernel_initializer='uniform')(L8)
    #L10 = BatchNormalization()(L9)
    L10 = Dense(units=128,activation='relu',kernel_initializer='uniform')(L9)
    L11 = Dense(units=128,activation='relu',kernel_initializer='uniform')(L10)
    L12 = Add()([L9,L11])
    L17 = Dense(units=Y_train.shape[1],activation='sigmoid')(L12)
    dnn_model = tensorflow.keras.Model(inputs=input_layer, outputs=L17)
    
    return dnn_model

