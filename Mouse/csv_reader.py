import pandas as pd
import time
from ast import literal_eval
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import more_data
from sklearn.utils import shuffle

def make_array_from_data(data):
    array=data.values
    for i in range(data.shape[0]):
        array[i][4]=literal_eval(array[i][4])
    return array

def create_data(array):
    All_data=[]
    for row in array:
        data=[]
        x=row[2]-row[0]
        y=row[3]-row[1]
        data.append(x)
        data.append(y)
        data.extend(row[4])
        All_data.append(data)
    return All_data

def plot_path(path):
    x=[]
    y=[]
    x_sum=0
    y_sum=0
    for i in range(2,len(path)-1,2):
        x.append(x_sum)
        y.append(y_sum)
        x_sum+=path[i]
        y_sum+=path[i+1]
        
    plt.plot(x,y)
    plt.title("Go To:  "+str(path[0])+","+str(path[1]))
    plt.xlim(-1920, 1920)
    plt.ylim(-1080, 1080)
    plt.gca().invert_yaxis()
    plt.show()

def add_tuple_to_array(array):
    tuppled=[]
    for i in range(0,len(array)-1,2):
        cord=(array[i],array[i+1])
        tuppled.append(cord)
    return tuppled

def final_array(array):
    final=[]
    for line in array:
        final.append(add_tuple_to_array(line))
    return final

def Create_Dataset(data):
    dataset=[]
    for path in data:
        c_path=path.copy()
        end=[c_path[0][0],c_path[0][1]]
        c_path.remove(c_path[0])
        inputs = np.zeros([len(c_path), 2])
        inputs[0] = [end[0], end[1]]
        for i in range(len(c_path)-1):
            end[0] = end[0] - c_path[i][0]
            end[1] = end[1] - c_path[i][1]
            inputs[i+1] = [end[0],end[1]]
        labels = np.zeros([len(c_path), 2])
        for i in range(len(c_path)):
            labels[i] = [c_path[i][0],c_path[i][1]]
        inputs = torch.from_numpy(inputs).float()
        labels = torch.from_numpy(labels).float()
        dataset.append((inputs, labels))
    return dataset

def trim_zeros_from_back(x):
    for path in x:
        i=len(path)-1
        while i>=0 and (path[i][0]==0 and path[i][1]==0):
            del path[i]
            i-=1
    return x



def remove_zeros_completly(x):
    for path in x:
        i=len(path)-1
        while i>=0 and (path[i][0]==0 and path[i][1]==0):
            path.remove(path[i])
            i-=1
    return x
s=time.time()
data=pd.read_csv('trajectory_file.csv')
array=make_array_from_data(data)
data=create_data(array)

""" for i in range(1,10):
   plot_path(data[-i]) """
dataset=Create_Dataset(remove_zeros_completly(final_array(data)))
dataset.extend(more_data.path_dataset)
dataset=shuffle(dataset)
print("Eltelt idő: {} mp az adatok beolvasásával".format(round(time.time()-s,2)))
print("Minták száma: {}".format(len(dataset)))













