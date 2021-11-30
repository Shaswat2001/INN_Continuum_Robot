from sklearn.cluster import KMeans
from keras.initializers import Initializer

class KMeansCluster(Initializer):

    def __init__(self,X):
        self.X = X
    
    def __call__(self,shape,dtype=None,**kwargs):
        assert shape[1] == self.X.shape[1]
        n_clusters = shape[0]
        k_means = KMeans(n_clusters=n_clusters)
        k_means.fit(self.X)

        return k_means.cluster_centers_
    
