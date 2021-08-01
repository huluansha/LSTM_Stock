import train
from LSTM import LSTM
from data_loader import data_loader
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler

def main():
    tickers = ['AAPL']
    features = ['open','high','low','volume','closeadj']
    hidden_size, hidden_layer, lookback, epochs = 32, 2, 60, 100
    start_date, end_date = '2012-01-01', '2019-12-31'

    loader = data_loader(seq_len = lookback, select_features = features)
    model = LSTM(input_size = len(features), hidden_size = hidden_size, layer = hidden_layer)
    data = loader.load_data(tickers = tickers, start = start_date, end = end_date)
    output, hs = train.train_model(data, model, max_epochs = epochs)

    for ticker, ticker_data in zip(tickers, data):
        _, _, x, y, scaler = ticker_data
        hidden_state, cell_state = model.init_state(x.shape[0])
        y_predict, _ = model(x, (hidden_state, cell_state))
        price_scaler = MinMaxScaler()
        price_scaler.min_, price_scaler.scale_ = scaler.min_[-1], scaler.scale_[-1]
        actual, prediction = price_scaler.inverse_transform([torch.flatten(y).tolist()]), \
                             price_scaler.inverse_transform([torch.flatten(y_predict).tolist()])

        x_axis_val = range(actual.shape[1])
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x_axis_val, actual[0], color='tab:blue', label = 'actual')
        ax.plot(x_axis_val, prediction[0], color='tab:orange', label = 'prediction')
        plt.savefig("{}.png".format(ticker))
        plt.show(block=False)
        plt.pause(3)
        plt.close()




if __name__ == "__main__":
    main()