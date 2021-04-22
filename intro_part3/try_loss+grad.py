from difine_nn_S import *
from torch import optim
optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer.zero_grad()
output = model(images)
loss = criterion(output, labels)
loss.backward()
print(model[0].weight.grad)
optimizer.step()
print(model[0].weight)