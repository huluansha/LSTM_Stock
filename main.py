import train
from LSTM import LSTM
from data_loader import data_loader
import matplotlib.pyplot as plt
import torch
def main():
    batch_size = 4
    model = LSTM()
    loader = data_loader(batch_size = batch_size)

    train_x, train_y, test_x, test_y, scaler = loader.load_data(tickers = ['AAPL'],
                                start = '2020-01-02', end = '2021-07-01', numbers = 0)
    hidden_state, cell_state = train.train_model(train_x, train_y, model, max_epochs = 3, seq_length = loader.seq_len, batch_size = batch_size)

    actual, prediction = [], []
    x_axis_val = range(test_x.shape[0] * test_x.shape[1])
    for x, y in zip(test_x, test_y):
        y_predict, (hidden_state, cell_state) = model(x, (hidden_state, cell_state))
        actual.extend(torch.flatten(y).tolist())
        prediction.extend(torch.flatten(y_predict).tolist())

    plt.plot(x_axis_val, actual, label = 'actual')
    plt.plot(x_axis_val, prediction, label = 'predict')
    plt.savefig("aapl.png")
    plt.legend()
    plt.show()
    plt.close()



if __name__ == "__main__":
    main()