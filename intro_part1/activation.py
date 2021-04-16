import torch

# create activation function(Sigmoid)
def activation(x):
    return 1/ (1 + torch.exp(-x))