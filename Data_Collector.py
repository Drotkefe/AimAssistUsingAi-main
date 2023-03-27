import pygame
import random
import pyautogui
import csv
import math
import time
import pandas as pd
pygame.init()

def get_distance(x,y):
    mouse_cord=pyautogui.position()
    return round(math.sqrt((x-mouse_cord[0])**2+(y-mouse_cord[1])**2),2)

screen = pygame.display.set_mode([1920, 1080])

header=['startx','starty','endx','endy','points']
f=open('trajectory_file.csv', 'a', newline='')
writer = csv.writer(f)
#writer.writerow(header)


def get_mouse_data():
    before=pyautogui.position()
    for i in range(20000):
        pass
    after=pyautogui.position()
    path.append((after[0]-before[0],after[1]-before[1]))


def trim(t):
    m=0
    t.pop(0)
    for i in range(len(t)-1):
        if (t[i-m][0]==0 and t[i-m][1]==0):
            t.remove(t[i-m])
            m+=1
        else:
            return t

def lesser_than_1000(t):
    return len(t)<=1000 if True else False

def make_it_1000_padding(t):
    for i in range(1000):
        if i > len(t)-1 and len(t)<1001:
            t.append((0,0))
    return t

def make_it_numbers(t):
    tomb=[]
    for i in range(0,len(t)-1):
        tomb.append(t[i][0])
        tomb.append(t[i][1])
    return tomb

szamol=False
running = True
path=[]
posx,posy=random.randint(5,1915),random.randint(5,1075)
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type==pygame.MOUSEBUTTONUP and get_distance(posx,posy) < 5:
            init_posx=posx
            init_posy=posy
            if szamol==False:
                szamol=True
                posx=random.randint(5,1915)
                posy=random.randint(5,1075)
                path=[]
                path.append((init_posx,init_posy))
            else:
                szamol=False
                print(len(path),end=' ')
                if (lesser_than_1000(path)):
                    s1=path[0][0]
                    s2=path[0][1]
                    rows=[s1,s2,posx,posy,make_it_numbers(make_it_1000_padding(trim(path)))]
                    writer.writerow(rows)
               
    screen.fill((255, 255, 255))

    pygame.draw.circle(screen, (0, 0, 255), (posx, posy), 5)
    if szamol:
        pygame.draw.circle(screen,(255,0,0), (1850,80),20)
        if get_distance(posx,posy) > 0.5:
            get_mouse_data()
        
    pygame.display.flip()
    

pygame.quit()
f.close()


