import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid
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

d=csv_reader.data_trimmed   #próbáljuk ki a csúnyábbik beolvasással, ahol még nem tuple-k
def create_labels(dataset):
    labels=[]
    for i in dataset:
        labels.append(i[0])
    return labels

labels=create_labels(d)

def create_data(dataset):
    data=[]
    for i in dataset:
        data.append(i[1:])
    return data

data=create_data(d)

dataset=[[labels[i], data[i]] for i in range(len(d))]


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1000, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, label):
        x=torch.cat([x,label],1)
        output = self.model(x)
        return output
    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(200, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1000),
            nn.Tanh()
        )

    def forward(self, x, labels):
        x=torch.cat([x,labels],1)
        output = self.model(x)
        return output


batch_size = 128  # Batch size
epochs = 300  # Train epochs
learning_rate = 0.0001

num_epochs = 300
loss_function = nn.BCELoss()
data_loader=torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

generator=Generator().to(device)
discriminator=Discriminator().to(device)

g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion):
    
    # Init gradient
    g_optimizer.zero_grad()
    
    # Building z
    z = Variable(torch.randn(batch_size, 200)).to(device)
    
    # Building fake labels
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 1920, batch_size))).to(device)
    
    # Generating fake data
    fake_data = generator(z, fake_labels)
    
    # Disciminating fake data
    validity = discriminator(fake_data, fake_labels)
    
    # Calculating discrimination loss (fake data)
    g_loss = criterion(validity, Variable(torch.ones(batch_size)).to(device))
    
    # Backword propagation
    g_loss.backward()
    
    #  Optimizing generator
    g_optimizer.step()
    
    return g_loss.data

def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_data, labels):
    
    # Init gradient 
    d_optimizer.zero_grad()

    # Disciminating real data
    real_validity = discriminator(real_data, labels)
    
    # Calculating discrimination loss (real data)
    real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).to(device))
    
    # Building z
    z = Variable(torch.randn(batch_size, 200)).to(device)
    
    # Building fake labels
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 1920, batch_size))).to(device)
    
    # Generating fake data
    fake_data = generator(z, fake_labels)
    
    # Disciminating fake data
    fake_validity = discriminator(fake_data, fake_labels)
    
    # Calculating discrimination loss (fake data)
    fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)).to(device))
    
    # Sum two losses
    d_loss = real_loss + fake_loss
    
    # Backword propagation
    d_loss.backward()
    
    # Optimizing discriminator
    d_optimizer.step()
    
    return d_loss.data

for epoch in range(epochs):
    
    print('Starting epoch {}...'.format(epoch+1))
    
    for i, (labels, data) in enumerate(data_loader):
        
        # Train data
        real_data = Variable(data[i][i])
        labels = Variable(labels[i])
        
        # Set generator train
        generator.train()
        
        # Train discriminator
        d_loss = discriminator_train_step(len(real_data), discriminator,
                                          generator, d_optimizer, loss_function,
                                          real_data, labels)
        
        # Train generator
        g_loss = generator_train_step(batch_size, discriminator, generator, g_optimizer, loss_function)
    
    # Set generator eval
    generator.eval()
    
    print('g_loss: {}, d_loss: {}'.format(g_loss, d_loss))
    
    # Building z 
    z = Variable(torch.randn(1920, 200)).to(device)
    
    # Labels 0 ~ 8
    labels = Variable(torch.LongTensor(np.arange(1920))).to(device)
    
    # Generating data
    sample_data = generator(z, labels).unsqueeze(1).data.cpu()
    