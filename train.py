import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from LSTM import LSTM
import data_loader


def train_model(loader, model, max_epochs, seq_length):
    model.train()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    data = loader.load_data()
    for epoch in range(max_epochs):
        hidden_state, cell_state = model.init_state(1)

        for batch, (x, y) in data:
            optimizer.zero_grad()

            y_pred, (hidden_state, cell_state) = model(x, (hidden_state, cell_state))
            loss = criterion(y_pred, y)

            hidden_state = hidden_state.detach()
            cell_state = cell_state.detach()

            loss.backward()
            optimizer.step()

            print({'epoch': epoch, 'batch': batch, 'loss': loss.item()})
    return (hidden_state, cell_state)