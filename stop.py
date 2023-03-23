import pygame
import random
import pyautogui
import csv
import math
import time
from pynput.mouse import Listener
pygame.init()

def get_distance(x,y):
    mouse_cord=pyautogui.position()
    return round(math.sqrt((x-mouse_cord[0])**2+(y-mouse_cord[1])**2),2)

screen = pygame.display.set_mode([1920, 1080])

header=['distance','points']
f = open('trajectory_file.csv', 'a', newline='') 
writer = csv.writer(f)
writer.writerow(header)


szamol=False
running = True
posx,posy=52,187

def get_mouse_data():
    #print(get_distance(posx,posy))
    mouse_cord=pyautogui.position()
    time.sleep(0.01)
    get_relative_step(mouse_cord[0],mouse_cord[1])

def get_relative_step(x,y):
    mouse_cord=pyautogui.position()
    #print(mouse_cord[0]-x,mouse_cord[1]-y)
    path.append((mouse_cord[0]-x,mouse_cord[1]-y))

def trim(t):
    m=0
    t.pop(0)
    for i in range(len(t)-1):
        if (t[i-m][0]==0 and t[i-m][1]==0):
            t.remove(t[i-m])
            m+=1
        else:
            return t
        
def make_it_120_padding(t):
    for i in range(120):
        if i > len(t)-1:
            t.append((0,0))
    return t

def make_it_string(t):
    s=""
    s1=str(t[0][0])
    s2=str(t[0][1])
    s+=s1+","+s2+","
    for i in range(1,len(t)-2):
        s1=str(t[i][0])
        s2=str(t[i][1])
        s+=s1+","+s2+","
    s+=str(t[len(t)-1][0])+","+str(t[len(t)-1][1])
    return s 

path=[]
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type==pygame.MOUSEBUTTONUP:
            posx=random.randint(30,1820)
            posy=random.randint(30,870)
            if szamol==False:
                szamol=True
                path=[]
                path.append(get_distance(posx,posy))
            else:
                szamol=False
                rows=[[path[0],make_it_string(make_it_120_padding(trim(path)))]]
                writer.writerows(rows)
                #print(make_it_string(make_it_120_padding(trim(path))))
            
                
    # Fill the background with white
    screen.fill((255, 255, 255))

    pygame.draw.circle(screen, (0, 0, 255), (posx, posy), 5)
    if szamol:
        pygame.draw.circle(screen,(255,0,0), (1850,80),20)
        if get_distance(posx,posy) > 5:
            get_mouse_data()
        
    pygame.display.flip()

pygame.quit()
f.close()


