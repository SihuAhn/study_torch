import torch.nn as nn
import torch.nn.functional as F
import torch

# define the CNN architecture
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         # convolutional layer (sees 32x32x3 image tensor)
#         self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
#         # convolutional layer (sees 16x16x16 tensor)
#         self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
#         # convolutional layer (sees 8x8x32 tensor)
#         self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
#         # max pooling layer
#         self.pool = nn.MaxPool2d(2, 2)
#         # linear layer (64 * 4 * 4 -> 500)
#         self.fc1 = nn.Linear(64 * 4 * 4, 500)
#         # linear layer (500 -> 10)
#         self.fc2 = nn.Linear(500, 10)
#         # dropout layer (p=0.25)
#         self.dropout = nn.Dropout(0.25)
#
#     def forward(self, x):
#         # add sequence of convolutional and max pooling layers
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         # flatten image input
#         x = x.view(-1, 64 * 4 * 4)
#         # add dropout layer
#         x = self.dropout(x)
#         # add 1st hidden layer, with relu activation function
#         x = F.relu(self.fc1(x))
#         # add dropout layer
#         x = self.dropout(x)
#         # add 2nd hidden layer, with relu activation function
#         x = self.fc2(x)
#         return x
#
# # create a complete CNN
# model = Net()
# print(model)


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, 4)
        self.conv2 = nn.Conv2d(96, 256, 5, padding=2, groups=2)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, 3, padding=1, groups=2)
        self.conv5 = nn.Conv2d(384, 256, 3, padding=1, groups=2)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2, 2))

        # x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # x = F.max_pool2d(F.relu(self.conv5(x)), (2, 2))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, (2, 2))
        x = x.view(x.size(0), 256 * 6 * 6)
        x = F.dropout(F.relu(self.fc1(x)), p=0.5)
        x = F.dropout(F.relu(self.fc2(x)), p=0.5)
        x = self.fc3(x)
        return x


model = AlexNet()
train_on_gpu = torch.cuda.is_available()

# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()