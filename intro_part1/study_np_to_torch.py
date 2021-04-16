import numpy as np
import torch

np.set_printoptions(precision=3)
a = np.random.rand(4,3)
print(a)

b = torch.from_numpy(a)
print(b)
print(b.numpy())


print(b.mul(2))
print(a)