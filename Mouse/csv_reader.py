import pandas as pd
import time
from ast import literal_eval
import matplotlib.pyplot as plt

s=time.time()
data=pd.read_csv('trajectory_file.csv')
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
        x_sum+=path[i]
        y_sum+=path[i+1]
        x.append(x_sum)
        y.append(y_sum)

    plt.plot(x,y)
    plt.show()

array=make_array_from_data(data)
print(time.time()-s)
data=create_data(array)
for i in range(50):
    plot_path(data[len(data)-i-1])


#kölünbség 416 -213















""" points=array[-1]
i=2
#pyautogui.moveTo(array[-1][0],array[-1][1])
mouse=Controller()
print(array[-1][4])
print(len(array[-1][4]))
while i<len(array[-1][4]):
    x=array[-1][4][i]
    y=array[-1][4][i+1]
    mouse.move(x,y)
    for x in range(80000):
        pass
    i+=2
print(pyautogui.position())

print(array[-1][2],array[-1][3] ," kéne") """
