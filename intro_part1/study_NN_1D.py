from activation import *

# fix random number
torch.manual_seed(7)

# define features, weights, bias
features = torch.randn(1,5)
weights = torch.randn_like(features)
bias = torch.randn(1,1)
print(f'feature = {features}, \n weights = {weights}, \n bias = {bias}')

# Calculate the output of this network using the weights and bias tensors
output = activation(torch.sum(features * weights) + bias)
print(f'output(using sum & *) = {output}')

# Claculate the output of this network using matrix multiplication
output = activation(torch.mm(features, weights.T) + bias)
print(f'output(using matrix multiplication) = {output}')

