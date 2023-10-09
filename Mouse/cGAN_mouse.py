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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)

X=csv_reader.numpy_X  
Y=csv_reader.numpy_Y

def get_unique_labels(y):
    unique_data = [list(x) for x in set(tuple(x) for x in Y)]
    return len(unique_data)

class_numbers=get_unique_labels(Y)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(class_numbers, class_numbers)

        self.model = nn.Sequential(
            nn.Linear(999*2+class_numbers, 2048),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, label):
        x = x.view(-1, 999*2)
        c=self.label_emb(label)
        x=torch.cat([x,c],1)
        output = self.model(x)
        return output.squeeze()

z_size=200

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(class_numbers,class_numbers)
        self.model = nn.Sequential(
            nn.Linear(z_size+class_numbers, 512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(2048, 999*2),
            nn.Tanh()
        )

    def forward(self, z, labels):
        z=z.view(-1,z_size)
        c=self.label_emb(labels)
        z=torch.cat([z,labels],1)
        out = self.model(z)
        return out.view(-1,999*2)


batch_size = 128  # Batch size
epochs = 300  # Train epochs
learning_rate = 0.0001
num_epochs = 300
loss_function = nn.BCELoss()

dataset=[(X[i], Y[i]) for i in range(X.shape[0])]
data_loader=torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

generator=Generator().to(device)
discriminator=Discriminator().to(device)


g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion):
    
    # Init gradient
    g_optimizer.zero_grad()
    
    # Building z
    z = Variable(torch.randn(batch_size, z_size)).to(device)
    
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
    
    for i,(data, labels) in enumerate(data_loader):
        
        # Train data
        real_data = Variable(data).to(device)
        labels = Variable(labels[i]).to(device)
        
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
    