import cppimport.import_hook
import StaticModel
import random
import pickle

if __name__=="__main__":

    x = [10,10,10,10]
    StaticModel.solveModel(x[0],x[1],x[2],x[3])
    y = [float(i.strip()) for i in open("endeffector.dat").readlines()]
    y = x+y
    print(y)
    print(y[4:7])
    # points = []
    # for i in range(10000):
    #     x = [random.randint(5, 15) for p in range(4)]
    #     StaticModel.solveModel(x[0],x[1],x[2],x[3])
    #     y = [float(i.strip()) for i in open("endeffector.dat").readlines()]
    #     points.append(y)

    # f = open("Results/trajectory.pkl","wb")
    # # f = open("Results/"+args.Al   gorithm + ".pkl","wb")
    # pickle.dump(points,f)
    # f.close()