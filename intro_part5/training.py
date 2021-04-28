from data_load import *
from model import *

model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

epochs = 5
steps = 0

train_losses, test_losses = [], []
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()

        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        x, y = next(iter(testloader))
        ps = torch.exp(model(x))
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == y.view(*top_class.shape)

        accuracy = torch.mean(equals.type(torch.FloatTensor))
        print(f'Accuracy: {accuracy.item()*100}%')

torch.save(model.state_dict(), 'checkpoint.pth')