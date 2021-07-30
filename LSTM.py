import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self, input_size = 5, hidden_size = 128):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = 3

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first = True,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x, hidden_state):
        output, hidden_state = self.lstm(x, hidden_state)
        ret = self.fc(output)
        return ret.view(x.shape[1], -1)[-1], hidden_state

    def init_state(self, batch_size = 1):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))