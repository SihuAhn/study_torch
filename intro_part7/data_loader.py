import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms

import helper

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor()])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()])

torch.manual_seed(777)
train_set = datasets.ImageFolder('D:/code/study_torch/data/Cat_Dog_data/train/', transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

test_set = datasets.ImageFolder('D:/code/study_torch/data/Cat_Dog_data/test/', transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)

# def temp(loader):
#     data_iter = iter(loader)
#
#     images, labels = next(data_iter)
#     fig, axes = plt.subplots(figsize=(10,4), ncols=4)
#     for ii in range(4):
#         ax = axes[ii]
#         helper.imshow(images[ii], ax=ax, normalize=False)
#
# temp(train_loader)
# temp(test_loader)
#
# plt.show()