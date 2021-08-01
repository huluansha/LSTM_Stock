import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layer, bidirectional):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = layer
        self.bidirection = bidirectional

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first = True,
            bidirectional = self.bidirection,
            bias=False,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.hidden_size * (1+int(self.bidirection)), 1)

    def forward(self, x, hidden_state):
        output, hidden_state = self.lstm(x, hidden_state)
        ret = self.fc(output[:, -1, :])
        return ret, hidden_state

    def init_state(self, batch_size):
        return (torch.zeros(self.num_layers * (1+int(self.bidirection)), batch_size, self.hidden_size).requires_grad_(),
                torch.zeros(self.num_layers * (1+int(self.bidirection)), batch_size, self.hidden_size).requires_grad_())