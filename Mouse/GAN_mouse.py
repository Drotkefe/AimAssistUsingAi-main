import torch
from torch import nn

import math
import matplotlib.pyplot as plt
import csv_reader

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

dataset=csv_reader.data_trimmed

torch.manual_seed(111)

def create_train_data(dataset):
    train_data_length = len(dataset)
    train_set=[]
    for i in range(train_data_length):
        data=[]
        path=[]
        data.append(dataset[i][0])
        for j in range(1,len(dataset[i])):
            path.append(dataset[i][j])
            train_set.append([data,path])
    return train_set

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output
    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        output = self.model(x)
        return output

lr = 0.001
num_epochs = 300
loss_function = nn.BCELoss()

def main():
    train_set=create_train_data(dataset)
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    generator = Generator()
    discriminator = Discriminator()
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)
    

main()