from sklearn.cluster import KMeans
from keras.initializers import Initializer,Constant
from keras.engine.topology import Layer
import keras.backend as K
import numpy as np

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
    
    def build(self,input_dim):

        self.centers = self.add_weight("centers",shape=(self.output_dim,input_dim),initializer=self.initializer,trainable=True)
        self.betas = self.add_weight("betas",shape=(self.output_dim,),initializer=Constant(value=self.betas),trainable=True)

        super(RBFLayer,self).build(input_dim)
    
    def call(self,X):
        CTR = K.expand_dims(self.centers)
        Distance = K.sum(K.transpose(X)-CTR,axis=1)
        return K.exp(K.transpose((-self.betas*Distance)))
    
    def compute_output_shape(self,input_dim):
        return(self.output_dim,)

