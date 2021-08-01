import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from LSTM import LSTM
import data_loader


def train_model(data, model, max_epochs, rate, batch_size = None, shuffle = False):
    model.train()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=rate)
    print(model.parameters)
    for epoch in range(max_epochs):
        for ticker_data in data:
            x, y, _, _, _, _ = ticker_data
            if shuffle:
                idx = torch.randperm(x.size()[0])
                x, y = x[idx], y[idx]
            if not batch_size:
                batch_size = x.shape[0]
            i = 0
            batch = 1
            while i <= x.shape[0] - batch_size:
                hidden_state, cell_state = model.init_state(batch_size)
                optimizer.zero_grad()
                y_pred, (hidden_state, cell_state) = model(x[i:i+batch_size], (hidden_state, cell_state))
                loss = criterion(y_pred, y[i:i+batch_size])

                hidden_state = hidden_state.detach()
                cell_state = cell_state.detach()

                loss.backward()
                optimizer.step()
                i += batch_size

                print({'epoch': epoch, 'batch': batch, 'loss': loss.item()})
                batch += 1
    return model.eval(), hidden_state, cell_state