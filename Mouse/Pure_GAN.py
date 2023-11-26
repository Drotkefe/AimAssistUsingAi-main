import torch
import torch.nn as nn
import csv_reader
import matplotlib.pyplot as plt
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('torch version:',torch.__version__)
print('device:', device)

X=csv_reader.numpy_X

np.random.seed(12)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2*999, 999),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(999, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), 2*999)
        output = self.model(x)
        return output

discriminator = Discriminator().to(device=device)

z_size=300
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_size, 300),
            nn.ReLU(),
            nn.Linear(300, 450),
            nn.ReLU(),
            nn.Linear(450, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 2*999),
            nn.Tanh(),
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(x.size(0), 999, 2)
        return output
    
def plot_metrics(discriminator_loss,generator_loss):
    x=[i for i in range(1,len(dmetrics_loss)+1)]
    plt.plot(x,discriminator_loss,'r')
    plt.plot(x,generator_loss,'b')
    plt.xlabel('Training iteration')
    plt.ylabel('Loss')
    plt.show()

def get_path(sample):
    path=[]
    for i in range(0,222,3):
        x=0
        y=0
        for j in range(0,4,1):
            x+=int(sample[i+j][0]*5.6)
            y+=int(sample[i+j][1]*5.6)
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
        
    plt.plot(x,y,'bo-',markersize=5)
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

generator = Generator().to(device=device)

lr = 0.0001
num_epochs = 50
loss_function = nn.BCELoss()
batch_size = 512

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr/10)


data_loader=torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=False)

dmetrics_loss=[]
gmetrics_loss=[]

for epoch in range(num_epochs):
    for n, (real_samples) in enumerate(data_loader):
        # Data for training the discriminator
        real_samples = real_samples.to(device=device)
        real_samples_labels = torch.ones((real_samples.shape[0], 1)).to(
            device=device
        )
        latent_space_samples = torch.randn((real_samples.shape[0], z_size)).to(
            device=device
        )
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((real_samples.shape[0], 1)).to(
            device=device
        )
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat(
            (real_samples_labels, generated_samples_labels)
        )

        # Training the discriminator
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(
            output_discriminator, all_samples_labels
        )
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Data for training the generator
        latent_space_samples = torch.randn((real_samples.shape[0], z_size)).to(
            device=device
        )

        # Training the generator
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(
            output_discriminator_generated, real_samples_labels
        )
        loss_generator.backward()
        optimizer_generator.step()

        # Show loss
        dmetrics_loss.append(loss_discriminator.cpu().detach())
        gmetrics_loss.append(loss_generator.cpu().detach())

    print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
    print(f"Epoch: {epoch} Loss G.: {loss_generator}")
            # plot_metrics(dmetrics_loss,gmetrics_loss)
            # generated_samples=generated_samples.cpu().detach()   
            # path=get_path(generated_samples[0])    
            # plot_path(path)
            # a=input("Stop? y/n: ")
            # if a=='y':
            #     break
            



#plot_metrics(dmetrics_loss,gmetrics_loss)

latent_space_samples = torch.randn(batch_size, z_size).to(device=device)
generated_samples = generator(latent_space_samples)
#torch.save(generator,"models\gan_models\model.pth")
model_scripted = torch.jit.script(generator)
model_scripted.save('models\gan_models\proba2.pt') 


generated_samples = generated_samples.cpu().detach()


print(generated_samples[0])
for i in range(len(generated_samples)):    
    path=get_path(generated_samples[i])    
    plot_path(path)