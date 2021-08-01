import train
from LSTM import LSTM
from data_loader import data_loader
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
def main():
    stateless = False
    tickers = ['AAPL']
    #tickers = ['MSFT', 'AMZN', 'FB', 'GOOGL', 'GOOG', 'BRK.B', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V', 'UNH', 'HD', 'PG', 'MA', 'PYPL', 'DIS', 'ADBE', 'BAC']
    features = ['open','high','low','volume','closeadj']
    learning_rate, hidden_size, hidden_layer, lookback, epochs, split, batch_size, shuffle, bidirection, bias, predict_price \
                            = 0.01, 64, 2, 50, 100, 0.8, None, False, False, False, False
    start_date, end_date = '2012-01-01', '2019-12-31'

    loader = data_loader(seq_len = lookback, select_features = features)
    model = LSTM(input_size = len(features), hidden_size = hidden_size, layer = hidden_layer, bidirectional=bidirection)
    data = loader.load_data(tickers, start_date, end_date, split, predict_price)
    model, hidden_state, cell_state = train.train_model(data, model, max_epochs = epochs, rate=learning_rate, batch_size=batch_size, shuffle=shuffle)

    for ticker, ticker_data in zip(tickers, data):
        _, _, x, y, scaler, table = ticker_data
        if stateless:
            # hidden_state, cell_state = model.init_state(x.shape[0])
            hidden_state, cell_state = hidden_state[:, -1, :].view(hidden_layer*(1+int(bidirection)), 1, hidden_size), \
                                       cell_state[:, -1, :].view(hidden_layer*(1+int(bidirection)), 1, hidden_size)
            hidden_state, cell_state = hidden_state.repeat(1, x.shape[0], 1), cell_state.repeat(1, x.shape[0], 1)
            y_predict, _ = model(x, (hidden_state, cell_state))

        else:
            #hidden_state, cell_state = model.init_state(1)
            hidden_state, cell_state = hidden_state[:, -1, :].view(hidden_layer*(1+int(bidirection)), 1, hidden_size), \
                                       cell_state[:, -1, :].view(hidden_layer*(1+int(bidirection)), 1, hidden_size)
            y_predict = []
            for x_i in x:
                x_i = x_i.view(1, x_i.shape[0], x_i.shape[1])
                y_i, (hidden_state, cell_state) = model(x_i, (hidden_state, cell_state))
                y_predict.append(y_i)
            y_predict = torch.FloatTensor(y_predict)

        train_size = int(table.shape[0]*split)
        if not predict_price:
            base = x[:, -1, -1]
            y_predict = y_predict + base

        price_scaler = MinMaxScaler()
        price_scaler.min_, price_scaler.scale_ = scaler.min_[-1], scaler.scale_[-1]
        prediction = price_scaler.inverse_transform([torch.flatten(y_predict).tolist()])
        actual = [table["closeadj"][train_size:].values]
        prev_day_close = [table["closeadj"][train_size-1:-1].values]

        MSE_Test = sum([(x-y)**2 for x, y in zip(actual[0], prediction[0])])/len(x)
        MSE_Bench = sum([(x-y)**2 for x, y in zip(actual[0], prev_day_close[0])])/len(x)

        # Need to made this thing 2d for scaler, so need to do [0]
        print("Mean Square Error Test is {}".format(MSE_Test))
        print("Mean Square Error Benchmark is {}".format(MSE_Bench))

        textstr = '\n'.join((r'MSE_Test=%.4f' % (MSE_Test,),r'MSE_Bench=%.4f' % (MSE_Bench,)))

        fig = plt.figure(dpi=1200)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(table["date"][train_size:], actual[0], color='#C71585', linewidth=1, marker='.', label = 'actual', markeredgewidth=1,
         markersize=2)
        ax.plot(table["date"][train_size:], prediction[0], color='#00FA9A', linewidth=1, marker='.', label = 'prediction', markeredgewidth=1,
         markersize=2)
        ax.plot(table["date"][0:train_size], table["closeadj"][0:train_size], color='#6495ED', linewidth=1, label='historical')
        plt.ylabel('Adj Close')
        plt.xlabel('Time Step')
        plt.legend()
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        plt.savefig("{}.png".format(ticker))
        # plt.show(block=False)
        # plt.pause(3)
        # plt.close()




if __name__ == "__main__":
    main()