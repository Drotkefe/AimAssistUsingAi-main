import time
import math
import win32api
import cv2
import mss
import numpy as np
import torch
import dxcam
from threading import Thread
import ctypes


def Closest_enemy(list,body_multiplier,x,y):
    centers=[]
    distance=[]
    for i in list:
        if i[5]==0 and i[4]>0.3:
            width=i[2]-i[0]
            height=i[3]-i[1]
            center=(int(i[2]-width/2),int((i[3]-height*body_multiplier)))
            centers.append(center)
            distance.append(math.sqrt((center[0]-x/2)**2+(center[1]-y/2)**2))
    if len(centers)==0:
        return
    return centers[distance.index(min(distance,default=None))]
kernel32 = ctypes.windll.kernel32
kernel32.SetThreadPriority(kernel32.GetCurrentThread(), 25)
sqrt3 = np.sqrt(3)
sqrt5 = np.sqrt(5)
def wind_mouse(start_x, start_y, dest_x, dest_y,distance,t, G_0=20, W_0=5, M_0=3, D_0=15):
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
    while (dist:=np.hypot(dest_x-start_x,dest_y-start_y)) >= distance:
        W_mag = min(W_0, dist)
        if dist >= D_0:
            W_x = W_x/sqrt3 + (2*np.random.random()-1)*W_mag/sqrt5
            W_y = W_y/sqrt3 + (2*np.random.random()-1)*W_mag/sqrt5
        else:
            W_x /= sqrt3
            W_y /= sqrt3
            if M_0 < 3:
                M_0 = np.random.random()*2 + 1
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
            timer = kernel32.CreateWaitableTimerA(ctypes.c_void_p(), True, ctypes.c_void_p())
            delay = ctypes.c_longlong(t)
            kernel32.SetWaitableTimer(timer, ctypes.byref(delay), 0, ctypes.c_void_p(), ctypes.c_void_p(), False)
            kernel32.WaitForSingleObject(timer, 0xffffffff)
            win32api.mouse_event(0x0001,int(v_x),int(v_y))
            


def Aimbot(game,act_distance,mouse_speed,x,y,body_multiplier):
    if (torch.cuda.is_available()):
        print(torch.cuda.get_device_name(0))
    if(game=="Counter Strike: Global Offensive"):
        game="CS_GO_Modell.pt"
    else:
        game="Valorant.pt"
    model=torch.hub.load('ultralytics/yolov5','custom',path=game)
    x_plus=int((1920-x)/2)
    y_plus=int((1080-y)/2)
    region=(x_plus,y_plus,x_plus+x,y_plus+y)
    #camera=dxcam.create(region=region,output_color="BGRA")
    with mss.mss() as sct:
        monitor = {"top": y_plus, "left": x_plus, "width": x, "height": y}
    while True:
        last_time=time.time()
        img=np.array(sct.grab(monitor))
        #img=np.array(camera.grab(region=region))
        result=model(img)
        rl=result.xyxy[0].tolist()
        #print("felismerés ideje: {}".format(time.time() - last_time))
        if len(rl)>0:
            dest=Closest_enemy(rl,body_multiplier,x,y)
            if dest!=None and np.hypot(dest[0]-x/2,dest[1]-y/2):
                wind_mouse(x/2,y/2,dest[0],dest[1],distance=3,t=1,M_0=int(24*mouse_speed))
        #cv2.imshow('debug',np.squeeze(result.render())) 
        
        cv2.waitKey(-1)
        print("fps:",(1 / (time.time() - last_time)))
    camera.stop()  

Aimbot("Counter Strike: Global Offensive",500,1,640,300,0.85)