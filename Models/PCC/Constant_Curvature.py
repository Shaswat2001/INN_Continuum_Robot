import numpy as np
import math
import matplotlib.pyplot as plt
import random
import pandas as pd
import pickle

class ContinuumRobotPCC:

    def __init__(self):
        self.radius = 20
        self.noSegments = 1
        self.segLength = 340
        
    def actuator_to_configSpace(self,Tl):
        '''
            Perform self.transformation from actuator to configuration space for 4 tendon continuum robot
            
            Input: 
                Tl -- list consisting of tendon lengths
            Output:
                k -- Robot Curvature
                l -- Robot length
                phi -- Bending angle
        '''
        pi = math.pi
        if Tl[2]==Tl[0] and Tl[3]==Tl[1] and Tl[2]==Tl[3]:
            phi = 0
        elif Tl[2]==Tl[0] and Tl[3]==Tl[1]:
            phi = pi/2
            if Tl[3]>Tl[0]:
                phi = -pi/2
        elif Tl[2]==Tl[0]:
            phi = pi/2
            if Tl[1]>Tl[3]:
                phi = -pi/2
        else:
            phi = math.atan((Tl[3]-Tl[1])/(Tl[2]-Tl[0]))
            if Tl[1]>Tl[3] and Tl[0]>Tl[2]:
                phi+=pi
            elif Tl[3]>Tl[1] and Tl[2]<Tl[0]:
                phi = pi - abs(phi)
                

        if Tl[2]==Tl[0] and Tl[3]==Tl[1] and Tl[2]==Tl[3]:
            k = 0
        elif Tl[3]==Tl[1] and Tl[2]==Tl[0]:
            k = (Tl[0]-3*Tl[1]+Tl[2]+Tl[3])*math.sqrt(2)/(self.radius*sum(Tl))
        elif Tl[3]==Tl[1]:
            k = (Tl[1]-3*Tl[0]+Tl[2]+Tl[3])*math.sqrt((Tl[3]-Tl[1])**2+(Tl[2]-Tl[0])**2)/(self.radius*sum(Tl)*(Tl[2]-Tl[0]))
        else:
            k = (Tl[0]-3*Tl[1]+Tl[2]+Tl[3])*math.sqrt((Tl[3]-Tl[1])**2+(Tl[2]-Tl[0])**2)/(self.radius*sum(Tl)*(Tl[3]-Tl[1]))
        
        l = sum(Tl)/4
        return k,l,phi

    def rotation(self,ax=[0,0,1],tht=0):
        [x,y,z] = ax
        c = math.cos(tht)
        s = math.sin(tht)
        R = [[(1-c)*x**2+c,(1-c)*x*y-s*z,(1-c)*x*z+s*y],
            [(1-c)*x*y+s*z,(1-c)*y**2+c,(1-c)*z*y-s*x],
            [(1-c)*x*z-s*y,(1-c)*z*y+s*x,(1-c)*z**2+c]]
        R= np.array(R)
        return R

    def transform(self,rot,trans):
        Tr = np.append(rot,trans,axis=1)
        Tr = np.append(Tr,[[0,0,0,1]],axis=0)
        return Tr
    
    def config_to_taskSpace(self,k,l,phi):
    
        p0 = [[0],[0],[0]]
        if k == 0:
            p = [[0],[0],[l]]
        else:
            p = [[(1-math.cos(k*l))/k],[0],[math.sin(k*l)/k]]

        Tbr = self.transform(self.rotation(tht=phi),p0)
        theta = k*l
        Trt = self.transform(self.rotation(ax=[0,1,0],tht=theta),p)
        Ttm = self.transform(self.rotation(tht=-phi),p0)
        
        Tf = np.matmul(Tbr,Trt)
        Tf = np.matmul(Tf,Ttm)
        return Tf

    def rotation_quaternion(self,R):
    
        tr = np.trace(R)
        
        if tr>0:
            S = math.sqrt(tr+1)*2
            qw = 0.25 * S
            qx = (R[2][1] - R[1][2])/S
            qy = (R[0][2] - R[2][0])/S 
            qz = (R[1][0] - R[0][1])/S
        elif (R[0][0] > R[1][1]) and (R[0][0] > R[2][2]):
            S = math.sqrt(1+R[0][0]-R[1][1]-R[2][2])*2
            qw = (R[2][1] - R[1][2])/S
            qx = 0.25 * S
            qy = (R[0][1] + R[1][0])/S
            qz = (R[0][2] + R[2][0])/S 
        elif (R[1][1] > R[2][2]):
            S = math.sqrt(1+R[1][1]-R[0][0]-R[2][2])*2
            qw = (R[2][0] - R[0][2])/S
            qx = (R[0][1] + R[1][0])/S
            qy = 0.25 * S
            qz = (R[1][2] + R[2][1])/S 
        else:
            S = math.sqrt(1+R[2][2]-R[0][0]-R[1][1])*2
            qw = (R[1][0] - R[0][1])/S
            qx = (R[0][2] + R[2][0])/S 
            qy = (R[2][1] + R[1][2])/S
            qz = 0.25 * S 
        
        return [qw,qx,qy,qz]

def create_workspace():

    position=[]
    orientation = []
    config=[]
    pi = math.pi
    robot = ContinuumRobotPCC()
    inputLength = []

    for _ in range(10000):
        length = [random.randint(robot.segLength-30, robot.segLength+30) for _ in range(4)]
        k,l,phi = robot.actuator_to_configSpace(length)
        if abs(k*l)>pi/2:
            continue
        T = robot.config_to_taskSpace(k,l,phi)
        quat = robot.rotation_quaternion(T[0:3,0:3])
        orientation.append(quat)
        inputLength.append(length)
        position.append(T[:3,-1])

    inputLength = np.array(inputLength)
    position = np.array(position)
    orientation = np.array(orientation)

    return orientation,inputLength,position

def plot_workspace(position):
    xdata = []
    ydata = []
    zdata = []
    for i in range(len(position)):
        if position[i][2]<400:
            xdata.append(position[i][0])
            ydata.append(position[i][1])
            zdata.append(position[i][2])

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(xdata, ydata, zdata, c=zdata)
    plt.savefig('../../Results/Workspace/PCC_workspace.png')
    plt.close()


if __name__ == "__main__":

    orientation,inputLength,position = create_workspace()
    plot_workspace(position)
    arr = np.concatenate((inputLength,position,orientation),axis=1)
    f = open("../../Dataset/PCC.pkl","wb")
    pickle.dump(arr,f)
    f.close()
        