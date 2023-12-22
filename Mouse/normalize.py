import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import win32api

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(10)

def transform(data,new_x,new_y):
    sum_x=sum(data[::2])
    sum_y=sum(data[1::2])
    scalarx=new_x/sum_x
    scalary=new_y/sum_y
    for i in range(0,len(data),2):
        data[i]=int(data[i]*scalarx)
        data[i+1]=int(data[i+1]*scalary)
    return data

def plot_path(path,diffx,diffy):
    x=[]
    y=[]
    x_sum=0
    y_sum=0
    for i in range(0,len(path)-1,2):
        x.append(x_sum)
        y.append(y_sum)
        x_sum+=path[i]
        y_sum+=path[i+1]
        
    plt.plot(x,y,'bo-',markersize=4)
    plt.title("GAN generated mouse movement from (0,0) to ("+str(x_sum+diffx)+","+str(y_sum+diffy)+")",fontsize=25)
    plt.xlim(-1920, 1920)
    plt.ylim(-1080, 1080)
    plt.ylabel("Y",fontsize=18)
    plt.xlabel("X",fontsize=18)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.gca().invert_yaxis()
    plt.plot(x[0], y[0], 'or', markersize=9)
    plt.plot(x[-1], y[-1], 'or', markersize=9)
    #plt.autoscale(True,'both',True)
    plt.show()

def multiple_path_to_plot(paths):
    for path in paths:
        x=[]
        y=[]
        x_sum=0
        y_sum=0
        for i in range(0,len(path)-1,2):
            x.append(x_sum)
            y.append(y_sum)
            x_sum+=path[i]
            y_sum+=path[i+1]
        plt.plot(x,y,'bo-',markersize=4)
        #plt.plot(x[0], y[0], 'or', markersize=7)
        plt.plot(x[-1], y[-1], 'or', markersize=7)
    plt.xlim(-1920, 1920)
    plt.ylim(-1080, 1080)
    plt.ylabel("Y",fontsize=18)
    plt.xlabel("X",fontsize=18)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.gca().invert_yaxis()
    plt.show()
    

def apply_transform(x, scale_x,scale_y, rotate_angle):
    s1=time.time()
    for i in range(0,len(x),2):
        x[i]=x[i]* math.cos(rotate_angle) - x[i+1] * math.sin(rotate_angle)
        x[i+1]=x[i+1]* math.cos(rotate_angle) - x[i+1] * math.sin(rotate_angle)
        x[i]=x[i]*scale_x
        x[i+1]=x[i+1]*scale_y
    return x

def get_path(sample):
    path=[]
    for i in range(0,125,2):
        path.append(int(sample[i][0]*10.65))
        path.append(int(sample[i][1]*10.65))
    return path

#god_mode2 uses noise dim of 300 otherwise 100
model = torch.jit.load('C:/Users/User/Desktop/AimAssistUsingAi-main/Mouse/models/gan_models/lajos4.pt')
model.eval()

def Generate_gan_mouse_movement(new_x,new_y):
    with torch.no_grad():
        generated_samples = model(torch.randn(1, 100).to(device=device))
        generated_samples = generated_samples.cpu().detach()
        #plot_path(get_path(generated_samples[0]),0,0)
        path = transform(get_path(generated_samples[0]),new_x,new_y)
        diff_x=new_x-sum(path[::2])
        diff_y=new_y-sum(path[1::2])
        return path,diff_x,diff_y 

def Test_Move(a):
    for i in range(0,len(a)-1,2):
        win32api.mouse_event(0x0001,a[i],a[i+1],0,0)
        for k in range(50000):
            pass

def produce_path_for_pure_results(sample):
    path=[]
    for i in range(0,222,3):
        x=0
        y=0
        for j in range(0,4,1):
            x+=int(sample[i+j][0]*5.6)
            y+=int(sample[i+j][1]*5.6)
        path.append(x)
        path.append(y)
    return path

if __name__ == "__main__":
    paths=[]
    with torch.no_grad():
        generated_samples = model(torch.randn(100, 300).to(device=device))
        generated_samples = generated_samples.cpu().detach()
        for i in range(0,len(generated_samples),10):
            path = get_path(generated_samples[i])
            paths.append(path)
        multiple_path_to_plot(paths)
        for i in range(len(generated_samples)):    
            path=produce_path_for_pure_results(generated_samples[i])    
            plot_path(path,0,0)
    #path,x,y=Generate_gan_mouse_movement(500,900)
    #plot_path(path,x,y)
    #Test_Move(path)
    

    #win32api.mouse_event(0x0001,x,y,0,0)
# plot_path(apply_transform(data[1646],100/170,100/166,math.radians(180)))

#main()