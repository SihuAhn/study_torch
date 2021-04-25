from model import model
from torch import optim, nn
from data_load import trainloader

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epoch = 5
for e in range(epoch):
    running_loss = 0
    for x_train, y_train in trainloader:
        x_train = x_train.view(x_train.shape[0], -1)
        optimizer.zero_grad()
        output = model(x_train)

        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss / len(trainloader)}")
