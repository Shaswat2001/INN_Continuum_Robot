import numpy as np
from main import build_parse
from Models.Static import StaticModel
import math
import pickle
import cv2
import os
import glob
import yaml
import matplotlib.pyplot as plt
from utils import denormalize_tension

num = 0

def ForwardKinematics(tension):
    StaticModel.solveModel(tension[0,0],tension[0,1],tension[0,2],tension[0,3])
    ee = [float(i.strip()) for i in open("endeffector.dat").readlines()]
    ee = np.array(ee[:3]).reshape(1,3)
    return ee

def getrobot(trajectory,pos):
    
    robotList = []
    global num
    for i in range(3):
        f = open("centerline.dat").readlines()[i].strip()
        f = " ".join(f.split())
        robotList.append([float(i) for i in f.split(" ")])
    robot = [[robotList[0][i],robotList[1][i],robotList[2][i]] for i in range(len(robotList[0]))]

    robot = np.array(robot,dtype=np.float64)*100
    fig = plt.figure()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(robot[:,0], robot[:,1], robot[:,2])    
    ax.scatter(trajectory[:,0]*100, trajectory[:,1]*100, trajectory[:,2]*100,c="red",s=1) 
    ax.plot(pos[:,0]*100, pos[:,1]*100, pos[:,2]*100,c="green")  
    ax.set_facecolor('white')
    plt.xticks(range(0,40,5))      
    plt.yticks(range(0,40,5))
    ax.view_init(elev = 23, azim = 45)
    plt.savefig("Results/Trajectory/"+args.learningModel+f"/ITR_{num}",transparent=True,bbox_inches='tight',pad_inches = 0)
    plt.close("all")
    num +=1

def plot_results(actual_pos,predicted_pos):
    
    fig = plt.figure()
    plt.scatter(actual_pos[:,0]*100, actual_pos[:,1]*100,c="Red",s=1)  
    plt.plot(predicted_pos[:,0]*100, predicted_pos[:,1]*100,c="green") 
    # plt.view_init(elev=10., azim=100)
    # plt.xticks(range(0,10,1))      
    # plt.yticks(range(0,10,1))
    plt.show()

def CircletrajectoryPosition():

    z = 0.397
    point_num = 100
    radius = 0.045
    points = []
    for i in range(point_num):
        cord = [radius*math.sin(2*i*math.pi/point_num)+0.2,radius*math.cos(2*i*math.pi/point_num)+0.2,z]
        points.append(cord)
    points.append(points[0])

    return points

def SquaretrajectoryPosition():

    z = 0.397
    point_num = 100
    radius = 0.045
    points = []
    for i in range(point_num):
        x = radius*math.cos(2*i*math.pi/point_num)
        y = radius*math.sin(2*i*math.pi/point_num)
        val = max(abs(x)/radius,abs(y)/radius)
        cord = [x/val+0.2,y/val+0.2,z]
        points.append(cord)
    points.append(points[0])

    return points

def generate_video(path):
    '''
    Generates video using the image saved
    '''
    img_array = []
    size=()
    # Loop though all the image files
    for filename in sorted(glob.glob(f'{path}/*.png'),key=os.path.getmtime):
        # read the image
        img = cv2.imread(filename)
        #dimension of the image
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter(f'{path}/project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15,size)

    for i in range(len(img_array)):
        #image added to the video writer
        out.write(img_array[i])
        
    out.release()

if __name__ == "__main__":
    
    args = build_parse()
    f = open("Results/Training/"+args.learningModel + ".sav",'rb')
    model = pickle.load(f)
    f.close()

    
    trajectory = CircletrajectoryPosition()
    rst = []
    tension_zero = np.array([10,10,10,10],dtype=np.float32).reshape(1,4)
    for curve in trajectory:
        curve = np.array(curve).reshape(1,3)
        tension_zero = model.predict(curve)
        tension_zero = denormalize_tension(tension_zero,5,15)
        result = ForwardKinematics(tension_zero)
        rst.append(result[0])
        getrobot(np.array(trajectory),np.array(rst))
    rst.append(rst[0])

    
    rst = np.array(rst)
    trajectory = np.array(trajectory)
    plot_results(trajectory,rst)

    f = open("Results/Trajectory/"+args.learningModel+"/circle.pkl","wb")
    pickle.dump(rst,f)
    f.close()

    generate_video("Results/Trajectory/"+args.learningModel)
    # print(trajectory)
    # print(rst)

