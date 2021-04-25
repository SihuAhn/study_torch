from training import model
from data_load import testloader

x_test, y_test = next(iter(testloader))

x_test = x_test.view(x_test.shape[0], -1)
y_pred = model(x_test)

for i in range(len(y_test)):
    print(y_test[i],(y_pred[i]==min(y_pred[i])).nonzero(as_tuple=True)[0])

