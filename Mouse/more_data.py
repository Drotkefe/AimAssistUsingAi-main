import torch
import numpy as np
import math


def xy_differ(x1, y1, x2, y2):
    if x1 >= 0 and x2 < 0:
        return True
    elif x1 <= 0 and x2 > 0:
        return True
    elif y1 >= 0 and y2 < 0:
        return True
    elif y1 <= 0 and y2 > 0:
        return True
    else:
        return False

def setup_path_dataset(dataset):
    data = []
    for path in dataset:
        f_path = path.copy()
        if len(path) > 0:
            end = [path[len(path)-1][0],path[len(path)-1][1],path[len(path)-1][2]]
            f_path.pop(len(path)-1)
        else:
            end = [path[0][0],path[0][1],path[0][2]]
        inputs = np.zeros([len(f_path), 2])
        inputs[0] = [end[0], end[1]]
        for i in range(len(f_path)-1):
            # Since we want next time to have been moved closer to our targeted point
            end[0] = end[0] - f_path[i][0]
            end[1] = end[1] - f_path[i][1]
            inputs[i+1] = [end[0],end[1]]
        labels = np.zeros([len(f_path), 2])
        for i in range(len(f_path)):
            labels[i] = [f_path[i][0],f_path[i][1]]
        inputs = torch.from_numpy(inputs).float()
        labels = torch.from_numpy(labels).float()
        data.append((inputs, labels))
    return data

def beolvasas():
    file = open("./data/1.5k_1ms.txt")
    dataset = []

    # Loop over the contents of the file, line by line
    for line in file:
        # Our dataset consists of each line containing one path, and each point in the path is a triplet of data
        line = line.replace("[","").replace("]","")
        path = []
        x, y, t = 0, 0, 0
        zeros = 0
        lines = line.split("),")
        
        # We loop over every triplet point in the current path
        for triplet in lines:
            numbs = (triplet.split("(")[1]).split(",")
            
            # We check if the triplet we are on is the last one, if it is then we can create the current move and add the size of the movement
            # We are required to do this due to each path always ends with a triplet that sums up the whole path, (x_total, y_total, time_total)
            if lines.index(triplet) == len(lines)-1:
                path.append((x,y,t))
                path.append((int(numbs[0]),int(numbs[1]),int(numbs[2].split(")")[0])))
                break
            
            curr_x = int(numbs[0])
            curr_y = int(numbs[1])
            
            # Counting up if we have stood still for too long, this doesnt always count consecutive halts 
            if curr_x == 0 and curr_y == 0:
                zeros += 1
            
            x += curr_x
            y += curr_y
            t += int(numbs[2].split(")")[0])
            
            # Check if the distance is met, if the direction is changed or if the point has stood still for too long 
            if math.sqrt((x**2)+(y**2)) > 3 or xy_differ(x, y, curr_x, curr_y) or zeros > 2:
                path.append((x, y, t))
                x, y, t, zeros = 0, 0, 0, 0
            
        dataset.append(path)
    return dataset

a=beolvasas()
path_dataset = setup_path_dataset(a)
#
#print("Number of paths: {}".format(len(dataset)))