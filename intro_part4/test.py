import torch
from torch import nn, optim
from torchvision import datasets, transforms


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

trainset = datasets.FashionMNIST('D:/code/study_torch/data/F_MNIST_data/', download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.FashionMNIST('D:/code/study_torch/data/F_MNIST_data/', download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

image, label = next(iter(trainloader))
print(f'image shape is : {image.shape}')
print(f'label shape is : {label.shape}')

model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

print(f'model architecture is : {model}')


criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epoch = 5
for e in range(epoch):
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1)
        optimizer.zero_grad()
        output = model(images)

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss = loss.item()
    else:
        print(f"Training loss: {running_loss / len(trainloader)}")

print()