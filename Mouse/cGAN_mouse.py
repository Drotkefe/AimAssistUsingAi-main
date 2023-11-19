import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import csv_reader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)

X=csv_reader.numpy_X  
Y=csv_reader.Labels

def get_unique_labels(y):
    unique_data = np.array(y)
    return len(np.unique(unique_data))

class_numbers=get_unique_labels(Y)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(class_numbers, class_numbers)

        self.model = nn.Sequential(
            nn.Linear(999*2+class_numbers, 1500),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1500, 1280),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1280, 1000),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1000, 784),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(784, 516),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(516, 320),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(320, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, label):
        x = x.view(-1, 999*2)
        c=self.label_emb(label)
        x=torch.cat([x,c],1)
        output = self.model(x)
        return output.squeeze()

z_size=100

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(class_numbers,class_numbers)
        self.model = nn.Sequential(
            nn.Linear(z_size+class_numbers, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 999*2),
            nn.Tanh()
        )

    def forward(self, z, labels):
        z=z.view(-1,z_size)
        c=self.label_emb(labels)
        x=torch.cat([z,c],1)
        out = self.model(x)
        return out.view(-1,999*2)


batch_size = 256  # Batch size
epochs = 20  # Train epochs
learning_rate = 0.0001
criterion = nn.BCELoss()

dataset=[(X[i], Y[i]) for i in range(X.shape[0])]
data_loader=torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

generator=Generator().to(device)
discriminator=Discriminator().to(device)

g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate/10)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

def get_path(sample):
    path=[]
    for i in range(0,222,3):
        x=0
        y=0
        for j in range(0,4,1):
            x+=int(sample[0][i+j]*5.6)
            y+=int(sample[0][i+j+1]*5.6)
        path.append(x)
        path.append(y)
    return path

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

def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion):
    
    # Init gradient
    g_optimizer.zero_grad()
    
    # Building z
    z = Variable(torch.randn(batch_size, z_size)).to(device)
    
    # Building fake labels
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, class_numbers, batch_size))).to(device)
    
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
    z = Variable(torch.randn(batch_size, z_size)).to(device)
    
    # Building fake labels
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, class_numbers, batch_size))).to(device)
    
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
        real_data = Variable(data).to(device)
        labels = Variable(labels).to(device)
        generator.train()
        d_loss = discriminator_train_step(len(real_data), discriminator,
                                          generator, d_optimizer, criterion,
                                          real_data, labels) 
        g_loss = generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion)
    generator.eval()
    print('g_loss: {}, d_loss: {}'.format(g_loss, d_loss))
    z = Variable(torch.randn(class_numbers, z_size)).to(device)
    labels = Variable(torch.LongTensor(np.arange(class_numbers))).to(device)
    # Generating data
    sample_data = generator(z, labels).unsqueeze(1).data.cpu()
    if (epoch+1)%10==0:
        # path=get_path(sample_data[1])
        # plot_path(path)
        print()

z = Variable(torch.randn(class_numbers, z_size)).to(device)
labels = Variable(torch.LongTensor(np.arange(class_numbers))).to(device)
# Generating data
sample_data = generator(z, labels).unsqueeze(1).data.cpu()
for i in range(0,3):
    path=get_path(sample_data[i])
    plot_path(path)