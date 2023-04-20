import time
from time import perf_counter
import math
import win32api
import cv2
import mss
import numpy as np
import torch
import dxcam
from threading import Thread


def Closest_enemy(list,body_multiplier,x,y):
    centers=[]
    distance=[]
    for i in list:
        if i[5]==0 and i[4]>0.8:
            width=i[2]-i[0]
            height=i[3]-i[1]
            center=(int(i[2]-width/2),int((i[3]-height*body_multiplier)))
            centers.append(center)
            distance.append(math.sqrt((center[0]-x/2)**2+(center[1]-y/2)**2))
    if len(centers)==0:
        return
    return centers[distance.index(min(distance,default=None))]

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
            start=time.time()
            #This should wait for the mouse polling interval
            #time.sleep(0.01)
            for i in range(t):
                pass
            #print(int(v_x),int(v_y))
            #print(time.time()-start)
            win32api.mouse_event(0x0001,int(v_x),int(v_y))
            #eger.move(int(v_x),int(v_y))
            step+=1
        step+=1

            
def mouse(rl,act_distance,body_multiplier,x,y,mouse_speed):
    if len(rl)>0:
        dest=Closest_enemy(rl,body_multiplier,x,y)
        if dest!=None and np.hypot(dest[0]-x/2,dest[1]-y/2) < act_distance:
            """ t1=create_thread(x/2,y/2,dest[0],dest[1],distance=2,t=0,G_0=20,W_0=5,M_0=12,D_0=15)
            t1.start() """
            wind_mouse(x/2,y/2,dest[0],dest[1],distance=2,t=0,M_0=int(2*mouse_speed))

def create_thread(start_x, start_y, dest_x, dest_y,distance,t,G_0=20, W_0=5, M_0=12, D_0=15):
    return Thread(target=wind_mouse,args=(start_x,start_y,dest_x,dest_y,distance,t,G_0, W_0, M_0, D_0))

fps=np.zeros(500)
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
    camera=dxcam.create(region=region,output_color="BGRA")
    with mss.mss() as sct:
        monitor = {"top": y_plus, "left": x_plus, "width": x, "height": y}
    i=0
    while True and i<500:
        last_time=perf_counter()
        #img=np.array(sct.grab(monitor))
        img=np.array(camera.grab(region=region)) #dxcamban dxcam duplicator.py és 0-at 100ra
        #last_time=perf_counter()
        result=model(img,size=320)
        rl=result.xyxy[0].tolist()
        #t1=Thread(target=mouse,args=(rl,act_distance,body_multiplier,x,y,mouse_speed))
        #last_time=perf_counter()
        mouse(rl,act_distance,body_multiplier,x,y,mouse_speed)
        #cv2.imshow('debug',np.squeeze(result.render())) 
        cv2.waitKey(0)
        print("fps:",1/(perf_counter() - last_time),end='\r')
        fps[i]=1 / (perf_counter() - last_time)
        i+=1
    camera.stop()  

Aimbot("Counter Strike: Global Offensive",1850,5,320,320,0.81)
print(np.average(fps))