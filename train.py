import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from LSTM import LSTM
import data_loader


def train_model(data, value, model, max_epochs, seq_length, batch_size):
    model.train()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(max_epochs):
        hidden_state, cell_state = model.init_state(batch_size)
        batch = 0
        for x, y in zip(data, value):
            optimizer.zero_grad()

            y_pred, (hidden_state, cell_state) = model(x, (hidden_state, cell_state))
            loss = criterion(y_pred, y)

            hidden_state = hidden_state.detach()
            cell_state = cell_state.detach()

            loss.backward()
            optimizer.step()

            print({'epoch': epoch, 'batch': batch, 'loss': loss.item()})
            batch += 1
    return (hidden_state, cell_state)