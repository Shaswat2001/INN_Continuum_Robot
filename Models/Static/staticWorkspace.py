import cppimport.import_hook
import StaticModel
import matplotlib.pyplot as plt
from numpy import random
import numpy as np
import pickle

def plot_workspace(position):
    xdata = []
    ydata = []
    zdata = []
    for i in range(len(position)):
        xdata.append(position[i][0])
        ydata.append(position[i][1])
        zdata.append(position[i][2])

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(xdata, ydata, zdata, c=zdata)
    plt.savefig('../../Results/Workspace/Static_workspace.png')
    plt.close()

if __name__=="__main__":


    points = []
    for i in range(10000):
        x = [5+10*random.rand() for p in range(4)]
        StaticModel.solveModel(x[0],x[1],x[2],x[3])
        y = [float(i.strip()) for i in open("endeffector.dat").readlines()]
        y = x+y
        points.append(y)

    points = np.array(points)
    plot_workspace(points[:,4:])
    f = open("../../Dataset/Static.pkl","wb")
    pickle.dump(points,f)
    f.close()