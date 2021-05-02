from data_loader import *
from model import *



model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

epochs = 5
steps = 0
device = 'cuda'
train_losses, test_losses = [], []
for e in range(epochs):
    running_loss = 0
    for images, labels in train_loader:
        images.to(device), labels.to(device)
        optimizer.zero_grad()

        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        test_loss = 0
        accuracy = 0

        with torch.no_grad() as no:
            model.eval()
            for images, labels in test_loader:
                images.to(device), labels.to(device)

                log_ps = model(images)
                test_loss += criterion(log_ps, labels)

                ps = torch.exp(model(images))
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        model.train().to(device)

        train_losses.append(running_loss/len(train_loader))
        test_losses.append(test_loss/len(test_loader))

        print(f'Epoch: {e+1}/{epochs}..\n'
              f'Training Loss: {train_losses[-1]:.3f}..\n'
              f'Test Loss: {test_losses[-1]:.3f}..\n'
              f'Test Accuracy: {accuracy/len(test_loader):.3f}..')

# import matplotlib.pyplot as plt
# plt.plot(train_losses, label='Training loss')
# plt.plot(test_losses, label='Validation loss')
# plt.legend(frameon=False)
# plt.show()

torch.save(model.state_dict(), 'checkpoint.pth')