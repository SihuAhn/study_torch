import torch.nn as nn
import torch.nn.functional as F
import torch



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
        x = F.relu(x)
        x = self.conv2(x)
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