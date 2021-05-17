from torch import nn


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim

        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):

        batch_size = x.size(0)

        r_out, hidden = self.rnn(x, hidden)
        r_out = r_out.view(-1, self.hidden_dim)

        output = self.fc(r_out)

        return output, hidden


