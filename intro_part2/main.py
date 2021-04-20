# import helper
from intro_part2.study_nn import Network_basic
from intro_part2.study_nn_F import Network
from study_nn_hardcoding import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

features = images.reshape(64, 784)
W1, W2, B1, B2 = multi_Layer_NW(features.shape[1], 256, 10)

# probabilities = calc_output(features, W1, W2, B1, B2)
# result = softmax(probabilities)
# print(result.sum(dim=1))
# print(probabilities.shape)
# print(result.max(dim=1))

# model = Network_basic()
# print(model.forward(features).max(dim=1))

model = Network()
model.hidden1.weight.data.normal_(std=0.01)
model.hidden1.bias.data.fill_(0)

dataiter = iter(trainloader)
images, labels = dataiter.next()
images.resize_(64, 1, 784)

img_idx = 0
ps = model.forward(images[img_idx, :])

img = images[img_idx]
print(img)
plt.imshow(images[1].view(28,28).numpy().squeeze(), cmap='Greys_r')
plt.show()
print(ps.shape)