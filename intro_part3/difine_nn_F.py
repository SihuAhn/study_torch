from data_loader import *

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.log_softmax(self.output(x), dim=1)

        return x


dataiter = iter(trainloader)
images, labels = next(dataiter)
images = images.view(images.shape[0], -1)

model = Network()
criterion = nn.CrossEntropyLoss()

logits = model(images)
loss = criterion(logits, labels)

print(loss)

