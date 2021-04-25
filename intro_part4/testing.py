from training import model
from data_load import testloader

x_test, y_test = next(iter(testloader))

x_test = x_test.view(x_test.shape[0], -1)
y_pred = model(x_test)
print(y_test[0], y_pred[0])

correct = 0
for i in range(len(y_test)):
    true = y_test[i]
    pred = (y_pred[i]==max(y_pred[i])).nonzero(as_tuple=True)[0][0]
    # print(true,pred)
    if true == pred:
        correct += 1

print(f'Accuracy is : {correct/len(y_test)}')
