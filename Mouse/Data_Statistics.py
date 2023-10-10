from csv_reader import data_trimmed,data
import matplotlib.pyplot as plt
import random

def avg_lenght_of_one_path():
    osszeg=0
    for i in data_trimmed:
        osszeg+=len(i)
    return osszeg/len(data_trimmed)

def avg_step_size():
    vegeredmeny=0
    for i in data_trimmed:
        osszeg=0
        kivonni_valo=0
        for j in range(1,len(i)):
            if(i[j][0]==0):
                kivonni_valo+=1
            else:
                osszeg+=abs(i[j][0])
        vegeredmeny+=osszeg/(len(i)-1-kivonni_valo)
    return vegeredmeny/len(data_trimmed)

def max_step_size():
    max_step=0
    for i in data_trimmed:
        for j in range(1,len(i)):
            if max_step< abs(i[j][0]):
                max_step=abs(i[j][0])
    return max_step

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
    plt.plot(x[0], y[0], 'or')
    plt.plot(x[-1], y[-1], 'or')
    plt.show()


def get_index(startx,starty):
    for i in range(len(data)):
        if data[i][0]==startx and data[i][1]==starty:
            return i

def longest_path():
    max=len(data_trimmed[0])
    for i in data_trimmed:
        if len(i)>max:
            max=len(i)
    return max

def main():
    print("Minták száma",len(data_trimmed))
    print("Átlagos lépésszám hossz:",avg_lenght_of_one_path())
    print("Legnagyobb lépés:",max_step_size())
    print("Átlagos pixel mérték lépésenként:",avg_step_size())
    print(longest_path())
    #plot_path(data[get_index(-1322,-561)])

    for i in range(15):
        plot_path(data[0])
    

if __name__ == '__main__':
    main()