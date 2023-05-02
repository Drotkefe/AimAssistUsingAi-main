import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import time
import matplotlib.pyplot as plt

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
            pred_path.append((int(np.round(pred_movement[0])), int(np.round(pred_movement[1]))))
        curr_loc[0] = curr_loc[0] + int(np.round(pred_movement[0]))
        curr_loc[1] = curr_loc[1] + int(np.round(pred_movement[1]))
        delta_loc = wanted_loc-curr_loc
        dist = np.sqrt(delta_loc[0]**2+delta_loc[1]**2)
        #If no more movement is generated stop generating movement
        if prev_dist == dist:
            is_stuck += 1
            if is_stuck > 2:
                break
        else:
            is_stuck = 0 
        # If moving the wrong way for too long stop generating movement
   
    pred_path.append((t[0], t[1]))
    #print("Path generated! {} movements needed to complete path".format(len(pred_path)-1))
    return pred_path

def plot_path(path):
    x=[]
    y=[]
    x_sum=0
    y_sum=0
    for i in range(len(path)-2):
        x.append(x_sum)
        y.append(y_sum)
        x_sum+=path[i][0]
        y_sum+=path[i][1]

    plt.plot(x,y)
    plt.title("Go To:  "+str(path[-1][0])+","+str(path[-1][1]))
    plt.xlim(-1920, 1920)
    plt.ylim(-1080, 1080)
    plt.gca().invert_yaxis()
    plt.show()

s1=time.time()
eger=generate_path(path,(0,0),(1874,500))
print("idő:",time.time()-s1)
plot_path(eger)
