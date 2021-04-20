import torch

x = torch.randn(1, 5)
print(x.size())
x = x.view(-1)
print(x.size())

x = torch.randn(2, 4)
print(x.size())
x = x.view(-1, 8)
print(x.size())

x = torch.randn(2, 4)
print(x.size())
x = x.view(-1)
print(x.size())

x = torch.randn(2, 4, 3)
print(x.size())
x = x.view(-1, 2)
print(x.size(), x)
x = x.view(2,4,3)
print(x.size())
x = x.reshape(-1, 2)
print(x.size(), x)
x = x.reshape(1, -1)
print(x.size(), x)
x = x.T
print(x.size(), x)