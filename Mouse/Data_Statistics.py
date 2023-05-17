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
    for i in range(2,len(path)-1,2):
        x.append(x_sum)
        y.append(y_sum)
        x_sum+=path[i]
        y_sum+=path[i+1]
        
    plt.plot(x,y,'bo-')
    plt.title("Human mouse movement from (0,0) to ("+str(path[0])+","+str(path[1])+")",fontsize=25)
    plt.xlim(-1920, 1920)
    plt.ylim(-1080, 1080)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.gca().invert_yaxis()
    plt.show()

def main():
    print("Minták száma",len(data_trimmed))
    print("Átlagos lépésszám hossz:",avg_lenght_of_one_path())
    print("Legnagyobb lépés:",max_step_size())
    print("Átlagos pixel mérték lépésenként:",avg_step_size())

    for i in range(10):
        plot_path(data[random.randint(0,1314)])
    

if __name__ == '__main__':
    main()