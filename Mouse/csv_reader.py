import pandas as pd
import time
from ast import literal_eval
import matplotlib.pyplot as plt
import copy
import math
import numpy as np
import torch
import more_data
from sklearn.utils import shuffle
import random

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
        
    plt.plot(x,y,linewidth=3)
    plt.title("Go To:  "+str(path[0])+","+str(path[1]),fontsize=25)
    plt.xlim(-1920, 1920)
    plt.ylim(-1080, 1080)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.gca().invert_yaxis()
    plt.show()

def add_tuple_to_array(array):
    tuppled=[]
    for i in range(0,len(array)-1,2):
        cord=(array[i],array[i+1])
        tuppled.append(cord)
    return tuppled

def add_array_to_array(array):
    new_array=[]
    for i in range(0,len(array)-1,2):
        cord=[array[i],array[i+1]]
        new_array.append(cord)
    return new_array

def final_array_tuple(array):
    final=[]
    for line in array:
        final.append(add_tuple_to_array(line))
    return final

def final_array_array(array):
    final=[]
    for line in array:
        final.append(add_array_to_array(line))
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

def remove_label(data):
    for d in data:
        for i in range(0,2):
            d.remove(d[0])

def get_numpy_labels(data):
    labels=[]
    for i in range(len(data)):
            labels.append([data[i][0],data[i][1]])
    return np.array(labels)

def get_lenghts(data):
    lenghts=[]
    for i in range(len(data)):
        x=math.sqrt(data[i][0]**2+data[i][1]**2)
        lenghts.append(x)
    return lenghts

s=time.time()
data_df=pd.read_csv('data/trajectory_file.csv')
array=make_array_from_data(data_df)
data=create_data(array)
print("Eltelt idő: {} mp az adatok beolvasásával".format(round(time.time()-s,2)))

numpy_Y=get_numpy_labels(data)
print("Cimkék:",numpy_Y.shape)

for i in range(1324,len(data)):
    if len(data[i])!=2000:
        for j in range(2000-len(data[i])):
            data[i].append(0)

lenghts=get_lenghts(data)
lenghts.sort()
lenghts=np.round(lenghts)

db_2=0
db_6=0
beyond=0
for i in range(0,len(lenghts)):
    if lenghts[i]<=200:
        db_2+=1
    elif lenghts[i]>200 and lenghts[i]<=600:
        db_6+=1
    else:
        beyond+=1
print("200-nál kisebb:",db_2,"\n200-600 között:",db_6,"\n600-on túl:    ",beyond)

remove_label(data)
a=np.array(data)
a=np.array(a.reshape(len(data),999,2))


numpy_X=a

data_array=final_array_array(data)
data_trimmed=final_array_tuple(data)
#print(data_trimmed[0])

dataset=Create_Dataset(remove_zeros_completly(final_array_tuple(data)))
dataset.extend(more_data.path_dataset)
dataset=shuffle(dataset)

def main():   
    print("Minták:",a.shape)

if __name__ == '__main__':
    main()
    















