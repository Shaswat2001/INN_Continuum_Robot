from CasCor import CasCor
import numpy  as np 
if __name__ == "__main__":
    X = np.array([[1,2,3],[2,3,4],[4,5,6]])
    Y = np.array([[1,2],[2,3],[3,4]])
    net = CasCor(X.shape[1],Y.shape[1])
    print(net.weights)
    Z,A = net.forward(X)
    print(Z)