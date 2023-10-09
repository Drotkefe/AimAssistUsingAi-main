import torch
from torch.nn.functional import normalize
from torchvision import transforms
# import csv_reader

# data=csv_reader.numpy_X

# t = torch.tensor(data,dtype=torch.float32)
# print("Tensor:", t[0][:50])
# t1=normalize(t,2.0,0)

# print(t1[0][:50])


# print(torch.rand(0,10,))

modell=torch.load("models\gan_models\model.pth")


