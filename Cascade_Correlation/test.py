from CasCor import CasCor
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
if __name__ == "__main__":
    data = pd.read_csv("MLDataset_quat.csv")
    dataS = data.drop('Unnamed: 0',axis=1)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(dataS)
    dataS = scaler.transform(dataS)
    X = dataS[:,0:7]
    Y = dataS[:,7:]
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,random_state=0)
    net = CasCor(X.shape[1],Y.shape[1])
    net.train(X_train,Y_train,X_test,Y_test)    
    print(net.train_loss)
    print("Done \n")
    print(net.test_loss)