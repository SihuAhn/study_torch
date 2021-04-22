from data_call import *



def activation(x):
    return 1/(1+torch.exp(-x))

def multi_Layer_NW(inputUnits, hiddenUnits, outputUnits):

    n_input = inputUnits
    n_hidden = hiddenUnits
    n_output = outputUnits

    W1 = torch.randn(n_input, n_hidden)
    W2 = torch.randn(n_hidden, n_output)

    B1 = torch.randn(1, n_hidden)
    B2 = torch.randn(1, n_output)

    return W1, W2, B1, B2

def calc_output(features, W1, W2, B1, B2):
    h = activation(torch.mm(features, W1) + B1)
    output = activation(torch.mm(h, W2) + B2)
    return output

def softmax(x):
    print(torch.sum(torch.exp(x), dim=1))
    print(torch.sum(torch.exp(x), dim=1).reshape(-1,2))
    print(torch.sum(torch.exp(x), dim=1).view(-1, 1))
    print(torch.sum(torch.exp(x), dim=1).view(1, -1).T)
    return torch.exp(x) / torch.sum(torch.exp(x), dim=1).view(-1, 1)

torch.manual_seed(7)
print(softmax(torch.randn([4,2])))
