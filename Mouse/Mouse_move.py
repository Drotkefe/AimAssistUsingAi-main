import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import time
import matplotlib.pyplot as plt
import win32api
import random

device = torch.device('cuda:0')

class PathNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
                        nn.Linear(2, 1000),
                        nn.LeakyReLU(),
                        nn.Linear(1000, 200),
                        nn.LeakyReLU(),
                        nn.Linear(200, 2)    
                    )
    def forward(self, x):
        return self.fc(x)



path=PathNet()
path.to(device)
path.load_state_dict(torch.load("./models/path_network_state_x"))

def generate_path(path_model, s, t):
    pred_path = []
    wanted_loc = np.array([t[0], t[1]])
    curr_loc = np.array([s[0], s[1]])
    is_stuck = 0
    prev_dist = 0

    delta_loc = wanted_loc-curr_loc
    dist = np.sqrt(delta_loc[0]**2+delta_loc[1]**2)
    # Looping new movement in the path while we have a distance from the target point
    while dist > 0:
        prev_dist = dist
        with torch.no_grad():
            inp = torch.from_numpy(delta_loc).float().to(device)
            pred_movement = path_model(inp)
            # If the function got supplied a model for generating time, we generate a time for this movement
            pred_movement = pred_movement.cpu().numpy()
            pred_path.append(int(np.round(pred_movement[0])))
            pred_path.append(int(np.round(pred_movement[1])))
        curr_loc[0] = curr_loc[0] + int(np.round(pred_movement[0]))
        curr_loc[1] = curr_loc[1] + int(np.round(pred_movement[1]))
        delta_loc = wanted_loc-curr_loc
        dist = np.sqrt(delta_loc[0]**2+delta_loc[1]**2)
        #If no more movement is generated stop generating movement
        if prev_dist == dist:
            is_stuck += 1
            if is_stuck > 10:
                break
        else:
            is_stuck = 0 
        # If moving the wrong way for too long stop generating movement
   
    pred_path.append(t[0])
    pred_path.append(t[1])
    #print("Path generated! {} movements needed to complete path".format(len(pred_path)-1))
    return pred_path

def plot_path(path,endx,endy,t):
    x=[]
    y=[]
    x_sum=0
    y_sum=0
    for i in range(2,len(path)-1,2):
        x.append(x_sum)
        y.append(y_sum)
        x_sum+=path[i]
        y_sum+=path[i+1]
        
    plt.plot(x,y,'bo-')
    plt.title("Regression modell results from (0,0) to ("+str(endx)+","+str(endy)+") \nsteps:"+str(len(path)//2)+", time needed: "+str(t)+"s" ,fontsize=25)
    plt.xlim(-1920, 1920)
    plt.ylim(-1080, 1080)
    plt.ylabel("Y",fontsize=18)
    plt.xlabel("X",fontsize=18)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.gca().invert_yaxis()
    plt.plot(x[0], y[0], 'or')
    plt.plot(x[-1], y[-1], 'or')
    plt.show()

def mouse_move(path):
    for i in path:
        win32api.mouse_event(0x0001,i[0],i[1])
        time.sleep(0.01)


for i in range(20):
    endx=random.randint(0,1421)
    endy=random.randint(0,874)
    s1=time.time()
    eger=generate_path(path,(0,0),(endx,endy))
    s2=np.round(time.time()-s1,3)
    plot_path(eger,endx,endy,s2)
#eger=generate_path(path,(0,0),(0,150))
print(len(eger))
print("idő:",time.time()-s1)
#mouse_move(eger[0:-2])
plot_path(eger)
