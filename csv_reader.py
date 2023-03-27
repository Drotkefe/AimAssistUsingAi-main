import pandas as pd
import win32api
import time
from ast import literal_eval

data=pd.read_csv('trajectory_file.csv')

def make_array_from_data(data):
    array=data.values
    for i in range(data.shape[0]):
        array[i][4]=literal_eval(array[i][4])
    return array

array=make_array_from_data(data)



