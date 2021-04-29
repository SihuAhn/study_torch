import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms

import helper

transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])

train_set = datasets.ImageFolder('D:/code/study_torch/data/Cat_Dog_data/train/', transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

test_set = datasets.ImageFolder('D:/code/study_torch/data/Cat_Dog_data/test/', transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

data_iter = iter(test_loader)

images, labels = next(data_iter)
fig, axes = plt.subplots(figsize=(10,4), ncols=4)
for ii in range(4):
    ax = axes[ii]
    helper.imshow(images[ii], ax=ax, normalize=False)

plt.show()