import pandas as pd
import win32api
import time
import pyautogui
from ast import literal_eval
from pynput.mouse import Controller

data=pd.read_csv('trajectory_file.csv')

def make_array_from_data(data):
    array=data.values
    for i in range(data.shape[0]):
        array[i][4]=literal_eval(array[i][4])
    return array

array=make_array_from_data(data)

points=array[-1]
i=2
pyautogui.moveTo(array[-1][0],array[-1][1])
time.sleep(1)
mouse=Controller()
print(array[-1][4])
print(len(array[-1][4]))
while i<len(array[-1][4]):
    x=array[-1][4][i]
    y=array[-1][4][i+1]
    mouse.move(x,y)
    time.sleep(0.001)
    i+=2
print(pyautogui.position())

print(array[-1][2],array[-1][3] ," kéne")
