import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5), (0.5,))])

trainset =  datasets.MNIST('D:/code/study_torch/data/MNIST_data', download=True, train=True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
