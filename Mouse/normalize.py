import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# data=csv_reader.numpy_X

# t = torch.tensor(data,dtype=torch.float32)
# print("Tensor:", t[0][:50])
# t1=normalize(t,2.0,0)

# print(t1[0][:50])


# print(torch.rand(0,10,))

# modell=torch.load("models\gan_models\model_200.pth")



def transform(data,new_x,new_y):
    sum_x=sum(data[::2])
    sum_y=sum(data[1::2])
    scalarx=new_x/sum_x
    scalary=new_y/sum_y
    for i in range(0,len(data),2):
        data[i]=int(data[i]*scalarx)
        data[i+1]=int(data[i+1]*scalary)
    print(sum(data[::2]))
    return data

def plot_path(path):
    x=[]
    y=[]
    x_sum=0
    y_sum=0
    for i in range(0,len(path)-1,2):
        x.append(x_sum)
        y.append(y_sum)
        x_sum+=path[i]
        y_sum+=path[i+1]
        
    plt.plot(x,y,'bo-')
    plt.title("Human mouse movement from (0,0) to ("+str(x_sum)+","+str(y_sum)+")",fontsize=25)
    plt.xlim(-1920, 1920)
    plt.ylim(-1080, 1080)
    plt.ylabel("Y",fontsize=18)
    plt.xlabel("X",fontsize=18)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.gca().invert_yaxis()
    plt.plot(x[0], y[0], 'or', markersize=12)
    plt.plot(x[-1], y[-1], 'or', markersize=12)
    plt.autoscale(True,'both',True)
    plt.show()

def apply_transform(x, scale_x,scale_y, rotate_angle):
    s1=time.time()
    for i in range(0,len(x),2):
        x[i]=x[i]* math.cos(rotate_angle) - x[i+1] * math.sin(rotate_angle)
        x[i+1]=x[i+1]* math.cos(rotate_angle) - x[i+1] * math.sin(rotate_angle)
        x[i]=x[i]*scale_x
        x[i+1]=x[i+1]*scale_y
    print("idő:", time.time()-s1)
    return x

def get_path(sample):
    path=[]
    for i in range(0,125,2):
        path.append(int(sample[i][0]*10.65))
        path.append(int(sample[i][1]*10.65))
    return path

model = torch.jit.load('C:/Users/User/Desktop/AimAssistUsingAi-main/Mouse/models/gan_models/lajos.pt')
model.eval()

def Generate_gan_mouse_movement(new_x,new_y):
    with torch.no_grad():
        generated_samples = model(torch.randn(1, 100).to(device=device))
        generated_samples = generated_samples.cpu().detach()
        return transform(get_path(generated_samples[0]),new_x,new_y)


def main():
    model = torch.jit.load('models\gan_models\lajos.pt')
    model.eval()

    with torch.no_grad():
        np.random.seed(12)
        for i in range(10):
            s1=time.time()
            generated_samples=model(torch.randn(1, 100).to(device=device))
            generated_samples = generated_samples.cpu().detach()
            print((time.time()-s1))
            plot_path(get_path(generated_samples[0]))

# plot_path(apply_transform(data[1646],100/170,100/166,math.radians(180)))

