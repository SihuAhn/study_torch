from torchvision import datasets, transforms
import torch

torch.manual_seed(7)
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

trainset = datasets.MNIST('D:/code/study_torch/data/MNIST_data', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# loop syntax
dataiter = iter(trainloader)
images, labels = dataiter.next()


# print(f'type of images = {type(images)}\n'
#       f'shape of images = {images.shape}\n'
#       f'shape of lables = {labels.shape}')

# plt.imshow(images[0].numpy().squeeze(), cmap='Greys_r')
# plt.show()