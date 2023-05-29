from customtkinter import *
import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
from Projectwork import Aimbot 
import multiprocessing

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

window=ctk.CTk()
window.geometry("1300x750")
window.title("Projectwork UI")
#window.resizable(0,0)
window.grid_columnconfigure(1, weight=2)

ctk.CTkLabel(master=window,text="Chose your game:",font = ("Arial", 30)).grid(row=0,column=0, sticky="N", padx=200, pady=50)
frame=ctk.CTkFrame(master=window, width=800, height=70).grid(row=0,column=1, pady=20)


#Aimbot(game,act_distance,mouse_speed,x,y,body_multiplier)
options=["Counter Strike: Global Offensive","Valorant"]
combobox_1=ctk.CTkComboBox(master=frame,width= 450,values=options,height=30)
combobox_1.grid(row=0, column=1)


bar_frames=ctk.CTkFrame(master=window, width=450, height=550,fg_color="#1f538d").grid(row=1,column=0, sticky="NWE", padx=80)

range=IntVar()
def act_slide_range(event):
    #act_value=tk.Label(master=bar_frames,text=int(act_range.get()),bg="#1a1a1a",fg="white").grid(column=0,row=1, sticky="NE", padx=200, pady=110)
    range.set(int(act_range.get()))

ctk.CTkLabel(master=bar_frames,text="Activation Distance:",bg_color="#1f538d",font=("Arial", 20),).grid(row=1,column=0, sticky="NW",padx=130,pady=50)
act_range=ctk.CTkSlider(master=bar_frames,width=300, from_= 100, to=1750, command=act_slide_range,number_of_steps=1650)
act_range.set(500)
range.set(int(act_range.get()))
act_range.grid(row=1, column=0, sticky="NW",padx=130,pady=90)
act_value=tk.Label(master=bar_frames,textvariable=range,bg="#1f538d",fg="white", font=("Arial",15)).grid(column=0,row=1, sticky="NE", padx=200, pady=105)



speed=StringVar()
def mouse_speed_slide(event):
    #act_value=tk.Label(master=bar_frames,text=int(act_range.get()),bg="#1a1a1a",fg="white").grid(column=0,row=1, sticky="NE", padx=200, pady=110)
    speed.set(str(round(speed_range.get(),2))+"x")

ctk.CTkLabel(master=bar_frames,text="Mouse Speed:",bg_color="#1f538d",font=("Arial", 20),).grid(row=1,column=0, sticky="NW",padx=130,pady=160)
speed_range=ctk.CTkSlider(master=bar_frames,width=300, from_= 0.2, to=5, command=mouse_speed_slide,number_of_steps=200)
speed_range.set(1.0)
speed.set(str(round(speed_range.get())) + "x")
speed_range.grid(row=1, column=0, sticky="NW",padx=130,pady=200)
speed_value=tk.Label(master=bar_frames,textvariable=speed,bg="#1f538d",fg="white",font=("Arial",15)).grid(column=0,row=1, sticky="NE", padx=200, pady=242)

size_x=StringVar()
size_y=StringVar()
def x_slide(event):
    #act_value=tk.Label(master=bar_frames,text=int(act_range.get()),bg="#1a1a1a",fg="white").grid(column=0,row=1, sticky="NE", padx=200, pady=110)
    size_x.set(str(int(size_x_range.get())))

def y_slide(event):
    size_y.set(str(int(size_y_range.get())))

ctk.CTkLabel(master=bar_frames,text="Detect Monitor Size:",bg_color="#1f538d",font=("Arial", 20),).grid(row=1,column=0, sticky="NW",padx=130,pady=270)
ctk.CTkLabel(master=bar_frames,text="X Cords:",bg_color="#1f538d",font=("Arial", 14),).grid(row=1,column=0, sticky="NW",padx=130,pady=320)
size_x_range=ctk.CTkSlider(master=bar_frames,width=300, from_= 256, to=1920, command=x_slide,number_of_steps=832, progress_color="#e40d11")
size_x_range.grid(row=1, column=0, sticky="NW",padx=130,pady=345)
size_x_range.set(320)
ctk.CTkLabel(master=bar_frames,text="Y Cords:",bg_color="#1f538d",font=("Arial", 14),).grid(row=1,column=0, sticky="NW",padx=130,pady=380)
size_y_range=ctk.CTkSlider(master=bar_frames,width=300, from_= 256, to=1080, command=y_slide,number_of_steps=412, progress_color="#e40d11")
size_y_range.grid(row=1, column=0, sticky="NW",padx=130,pady=405)
size_y_range.set(320)
size_x.set(str(int(size_x_range.get())))
size_y.set(str(int(size_y_range.get())))
size_x_value=tk.Label(master=bar_frames,textvariable=size_x,bg="#1f538d",fg="white",font=("Arial",15)).grid(column=0,row=1, sticky="NE", padx=200, pady=424)
size_y_value=tk.Label(master=bar_frames,textvariable=size_y,bg="#1f538d",fg="white",font=("Arial",15)).grid(column=0,row=1, sticky="NE", padx=200, pady=499)

target_frame=ctk.CTkFrame(master=window,height=550,width=520).grid(row=1,columnspan=3, sticky="NE", padx=80)

ctk.CTkLabel(master=target_frame,text="Select Target Zone:",font=("Arial", 20), bg_color="#212121").grid(row=1,columnspan=2, sticky="NE")
body=Image.open("Images/body_green_smal.png")
body=body.resize((275,570),Image.Resampling.LANCZOS)
body=ImageTk.PhotoImage(body)
ctk.CTkLabel(master=target_frame,image=body, text="").grid(row=1,columnspan=2,sticky="NE",pady=60,padx=80)

crosshair=Image.open("Images/crosshair.png")
crosshair=crosshair.resize((35,35),Image.Resampling.LANCZOS)
crosshair=ImageTk.PhotoImage(crosshair)
crosshair_pos=0
def move_crosshair(event):
    crosshair_pos=int(crosshair_range.get())
    show.grid_configure(pady=crosshair_pos)
    if crosshair_pos > 394:
        show.configure(bg="#212121")
    else:
        show.configure(bg="#1f8d23")
    
crosshair_range=ctk.CTkSlider(master=bar_frames,from_=610,to=80,orientation="vertical", height=460, command=move_crosshair)
crosshair_range.grid(row=1, columnspan=3,sticky="NE", pady=60, padx=200)
show=tk.Label(master=target_frame,image=crosshair,bg="#1f8d23",)
show.grid(row=1,columnspan=2, sticky="NE", padx=218, pady=int(crosshair_range.get()))

def return_run_values():
    print(combobox_1.get(),int(act_range.get()),round(speed_range.get(),2),int(size_x_range.get()),int(size_y_range.get()),((int(crosshair_range.get())-80)*-0.8)/530+0.9)
    return combobox_1.get(),int(act_range.get()),round(speed_range.get(),2),int(size_x_range.get()),int(size_y_range.get()),((int(crosshair_range.get())-80)*-0.8)/530+0.9



stop=False
def run_button_pressed():
    global stop
    global new_thread
    if(stop==False):
        new_thread=multiprocessing.Process(target=Aimbot,args=(return_run_values()))
    if(stop==False):
        new_thread.start()
        run_button.configure(text="Stop")
        stop=True
    elif(stop==True):
        run_button.configure(text="Run")
        new_thread.kill()
        stop=False

run_button=ctk.CTkButton(master=window,text="Run", height=50,command=run_button_pressed)
run_button.grid(row=0, column=2, padx=50)

if __name__ == "__main__":
    window.mainloop()


