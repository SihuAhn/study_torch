from model import *

fc_model.train(model, trainloader, testloader, criterion, optimizer, epochs=2)

print(f'Our mode: \n\n{model}\n')
print(f'The state dict keys: \n\n{model.state_dict().keys()}')

torch.save(model.state_dict(), 'our_model.pth')
