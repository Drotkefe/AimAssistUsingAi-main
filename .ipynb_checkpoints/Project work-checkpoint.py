
import time
import math
import win32api
import cv2
import mss
import numpy as np
import torch
from win32gui import FindWindow, GetWindowRect
import random
import threading
import multiprocessing


def get_window_geometry(name):
    window_handle = FindWindow(None, name)
    window_rect   = GetWindowRect(window_handle)
    return window_rect

x, y, w, h=get_window_geometry("Counter-Strike: Global Offensive - Direct3D 9")



def Closest_enemy(list):
    centers=[]
    distance=[]
    for i in list:
        if i[5]==0 and i[4]>0.7:
            width=i[2]-i[0]
            height=i[3]-i[1]
            center=(int(i[2]-width/2),int((i[3]-height*0.85)))
            centers.append(center)
            distance.append(math.sqrt((center[0]-w/2)**2+(center[1]-h/2)**2))
    if len(centers)==0:
        return
    return centers[distance.index(min(distance,default=None))]

sqrt3 = np.sqrt(3)
sqrt5 = np.sqrt(5)
def wind_mouse(start_x, start_y, dest_x, dest_y, G_0=9, W_0=2, M_0=3, D_0=12):
    '''
    WindMouse algorithm. Calls the move_mouse with each new step.
    Released under the terms of the GPLv3 license.
    G_0 - magnitude of the gravitational fornce
    W_0 - magnitude of the wind force fluctuations
    M_0 - maximum step size (velocity clip threshold)
    D_0 - distance where wind behavior changes from random to damped
    '''
    current_x,current_y = start_x,start_y
    v_x = v_y = W_x = W_y = 0
    while (dist:=np.hypot(dest_x-start_x,dest_y-start_y)) >= 1:
        W_mag = min(W_0, dist)
        if dist >= D_0:
            W_x = W_x/sqrt3 + (2*np.random.random()-1)*W_mag/sqrt5
            W_y = W_y/sqrt3 + (2*np.random.random()-1)*W_mag/sqrt5
        else:
            W_x /= sqrt3
            W_y /= sqrt3
            if M_0 < 3:
                M_0 = np.random.random()*3 + 3
            else:
                M_0 /= sqrt5
        v_x += W_x + G_0*(dest_x-start_x)/dist
        v_y += W_y + G_0*(dest_y-start_y)/dist
        v_mag = np.hypot(v_x, v_y)
        if v_mag > M_0:
            v_clip = M_0/2 + np.random.random()*M_0/2
            v_x = (v_x/v_mag) * v_clip
            v_y = (v_y/v_mag) * v_clip
        start_x += v_x
        start_y += v_y
        move_x = int(np.round(start_x))
        move_y = int(np.round(start_y))
        if current_x != move_x or current_y != move_y:
            #This should wait for the mouse polling interval
            for i in range(1500):
                pass
            win32api.mouse_event(0x0001,int(v_x),int(v_y))
            

def mouse_move_2(rl):
    dest=Closest_enemy(rl)
    if(dest!=None):
       dist=[int(dest[0])-int(w/2),int(dest[1])-int(h/2)]
       win32api.mouse_event(0x0001,dist[0],dist[1])

def mouse_move(dest): # ~0.22-0.13 sec
    dist=[int(dest[0])-int(w/2),int(dest[1])-int(h/2)]
    print("---")
    for i in range(1,50):
        print(dist,"distance")
        if ((math.sqrt(dist[0]**2+dist[1]**2))<5 ):
            break
        x=int((dist[0]*i/random.randint(50,60)))
        y=int((dist[1]*i/random.randint(50,60)))
        print(x,y)
        win32api.mouse_event(0x0001,x,y)
        dist[0]-=x
        dist[1]-=y
        for i in range(5000): #slow down between steps => time.sleep() is too slow 
            pass
    win32api.mouse_event(0x0001,dist[0],dist[1])

def mouse_move_1(dest):
    dist=[int(dest[0])-int(w/2),int(dest[1])-int(h/2)]
    print("---")
    for i in range(50):
        print(dist,"distance")
        if ((math.sqrt(dist[0]**2+dist[1]**2))<3 ):
            break
        if(dist[0]!=0 and dist[1]!=0):
            x=int(abs(dist[0])/dist[0])
            y=int(abs(dist[1])/dist[1])
        elif(dist[0]==0):
            x=0
            y=int(abs(dist[1])/dist[1])
        elif(dist[1]==0):
            x=int(abs(dist[0])/dist[0])
            y=0
        
        print(x,y)
        win32api.mouse_event(0x0001,x,y)
        dist[0]-=x
        dist[1]-=y
        

if (torch.cuda.is_available()):
    print(torch.cuda.get_device_name(0))
else:
    print("jó lesz neked a cpu is")


model=torch.hub.load('ultralytics/yolov5','custom',path='best.pt')

with mss.mss() as sct:
    monitor = {"top": y+300, "left": x+620, "width": 680, "height": 480}

w=680
h=480


def Aimbot():
    fps=[]
    while True:
        last_time=time.time()
        img=np.array(sct.grab(monitor))
        result=model(img)
        rl=result.xyxy[0].tolist()
        if len(rl)>0:
            dest=Closest_enemy(rl)
            if dest!=None and np.hypot(dest[0]-w/2,dest[1]-h/2) > 5:
                wind_mouse(w/2,h/2,dest[0],dest[1])
                #x=threading.Thread(target=wind_mouse,args=(w/2,h/2,dest[0],dest[1]))
                #x.start()

        
        #cv2.imshow('debug',np.squeeze(result.render()))
        #print("fps: {}".format(1 / (time.time() - last_time)))
        #cv2.waitKey(1)

Aimbot()




