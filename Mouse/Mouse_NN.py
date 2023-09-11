import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import datetime
import time
import csv_reader

dataset=csv_reader.dataset
print(len(dataset))

class PathNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
                        nn.Linear(2, 1000),
                        nn.LeakyReLU(),
                        nn.Linear(1000, 200),
                        nn.LeakyReLU(),
                        nn.Linear(200, 2)
                    )
    def forward(self, x):
        return self.fc(x)
    
device = torch.device('cuda:0')
loss_function = nn.MSELoss()

path_network = PathNet()
path_network.to(device)

path_opt = optim.Adam(path_network.parameters(), lr=0.0001)


def train(model, dataset, optimizer, loss_function, epochs=100):
    t = time.time()

    inf_found = False
    losses = []

    #Training the data
    for epoch in range(epochs):
        epoch_loss = []
        batch_nr = 0
        for inputs,labels in dataset:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            prediction = model(inputs) # Enter the i-th pos in the path into the neural network

            loss_t = loss_function(prediction, labels) # Our label will become the pos movement made by the premade dataset
            loss_t.backward()

            optimizer.step()     
            epoch_loss.append(loss_t.item())

            batch_nr = batch_nr + 1
            print(
                '\r[Train] Epoch {} [{}/{}] - Loss: {} \tProgress [{}%] \tTotal time elapsed: {}                                  '.format(
                    epoch+1, batch_nr, len(dataset), loss_t.item(), (epoch/epochs)*100, str(datetime.timedelta(seconds=round(time.time()-t)))
                ),
                end=''
            )

            if inf_found:
                break

        if inf_found:
            break
        losses.append(np.average(epoch_loss))

    print('\nTime elapsed from training: {}'.format(str(datetime.timedelta(seconds=round(time.time()-t)))))
    plt.plot(range(1,len(losses)+1),losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

train(path_network,dataset,path_opt,loss_function,50)
torch.save(path_network.state_dict(), "./models/path_network_large_data_x")
