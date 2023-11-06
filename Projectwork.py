from time import perf_counter
import math
import win32api
import cv2
import mss
import numpy as np
import torch
from threading import Thread
import psutil, os
import time
import sys
sys.path.insert(8,'C:/Users/User/Desktop/AimAssistUsingAi-main/Mouse')
from normalize import Generate_gan_mouse_movement



def Closest_enemy(list,body_multiplier,x,y):
    distance=[]
    for i in list:
        if i[5]==0 and i[4]>0.77:
            width=i[2]-i[0]
            height=i[3]-i[1]
            center=(int(i[2]-width/2),int((i[3]-height*body_multiplier)))
            distance.append((math.sqrt((center[0]-x/2)**2+(center[1]-y/2)**2),width+height,center))
    if len(distance)==0:
        return
    return sorted(distance,key=lambda x: (-x[1],x[0]))[0][2]

sqrt3 = np.sqrt(3)
sqrt5 = np.sqrt(5)
run=False
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
    global run
    step=0
    while (dist:=np.hypot(dest_x-start_x,dest_y-start_y)) >= distance and step < 15:
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
            #time.sleep(0.01)
            run=True
            for i in range(t):
                pass
            #print(int(v_x),int(v_y))
            #print(time.time()-start)
            win32api.mouse_event(0x0001,int(v_x),int(v_y))
            #eger.move(int(v_x),int(v_y))
            step+=1
        step+=1
    run=False

def mouse(rl,act_distance,body_multiplier,x,y,mouse_speed):
    if len(rl)>0:
        dest=Closest_enemy(rl,body_multiplier,x,y)
        if dest!=None and np.hypot(dest[0]-x/2,dest[1]-y/2) < act_distance:
            """ t1=create_thread(x/2,y/2,dest[0],dest[1],distance=2,t=0,G_0=20,W_0=5,M_0=12,D_0=15)
            t1.start() """
            wind_mouse(x/2,y/2,dest[0],dest[1],distance=3,t=int(5000),M_0=int(2*mouse_speed))

""" def create_thread(start_x, start_y, dest_x, dest_y,distance,t,G_0=20, W_0=5, M_0=12, D_0=15):
    return Thread(target=wind_mouse,args=(start_x,start_y,dest_x,dest_y,distance,t,G_0, W_0, M_0, D_0)) """


def Camera_Thread(x,y):
    x_plus=int((1920-x)/2)
    y_plus=int((1080-y)/2)

    monitor = {"top": y_plus, "left": x_plus, "width": x, "height": y}
    sct = mss.mss()
    global img
    img=np.array(sct.grab(monitor))
    while True:
        img=np.array(sct.grab(monitor))

""" def Mouse_Thread(body_multiplier,x,y,act_distance,mouse_speed):
    global rl
    while True:
        if len(rl)>0:
            dest=Closest_enemy(rl,body_multiplier,x,y)
            if dest!=None and np.hypot(dest[0]-x/2,dest[1]-y/2) < act_distance:
                t1=create_thread(x/2,y/2,dest[0],dest[1],distance=2,t=0,G_0=20,W_0=5,M_0=12,D_0=15)
                t1.start()
                wind_mouse(x/2,y/2,dest[0],dest[1],distance=5,t=0,M_0=int(1*mouse_speed)) """



def Aimbot(game,act_distance,mouse_speed,x,y,body_multiplier):
    if (torch.cuda.is_available()):
        print(torch.cuda.get_device_name(0))
    if(game=="Counter Strike: Global Offensive"):
        game="CS_GO_Modell.pt"
    else:
        game="Valorant.pt"
    
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)

    model=torch.hub.load('ultralytics/yolov5','custom',path=game)
    camera=Thread(target=Camera_Thread,args=(x,y))
    camera.start()
    time.sleep(0.5) # wait for camera relase one img
    while True:
        last_time=perf_counter()
        #img=np.array(camera.grab(region=region)) #dxcamban dxcam duplicator.py és 0-at 100ra
        result=model(img,size=x)
        rl=result.xyxy[0].tolist()
        #print("fps:",1/(perf_counter() - last_time),end='\r')
        #t1=Thread(target=mouse,args=(rl,act_distance,body_multiplier,x,y,mouse_speed))
        #last_time=perf_counter()
        if(run!=True):
            mouse(rl,act_distance,body_multiplier,x,y,mouse_speed)
        #cv2.imshow('debug',np.squeeze(result.render())) 
        cv2.waitKey(0)
        print("fps:",1/(perf_counter() - last_time),end='\r')


if __name__ == "__main__":

    Aimbot("Counter Strike: Global Offensive",1850,5,800,320,0.82)
