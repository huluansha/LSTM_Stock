
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import os

import train
from LSTM import LSTM
from data_loader import data_loader
from calculate_return import calculateReturn,buyAndHold
import sys
import warnings
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main():
    tickers = ['AAPL']
    tickers = ['MSFT', 'AMZN', 'FB', 'GOOGL', 'GOOG', 'BRK.B', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V', 'UNH', 'HD', 'PG', 'MA', 'PYPL', 'DIS', 'ADBE', 'BAC']
    features = ['open','high','low','volume','closeadj']
    count, learning_rate, weight_decay, hidden_size, hidden_layer, lookback, epochs, split, batch_size, stateless, shuffle, bidirection, bias, predict_price \
                            = 1, 0.01, 0, 64, 1, 60, 100, 0.80, 128, False, False, False, True, True
    # count, learning_rate, weight_decay, hidden_size, hidden_layer, lookback, epochs, split, batch_size, stateless, shuffle, bidirection, bias, predict_price \
    #      = arguments
    start_date, end_date = '2012-01-01', '2019-12-31'

    loader = data_loader(seq_len = lookback, select_features = features)
    model = LSTM(input_size = len(features) + 1 if not predict_price else len(features), hidden_size = hidden_size, layer = hidden_layer, bidirectional=bidirection)
    data = loader.load_data(tickers, start_date, end_date, split, predict_price)
    model, hidden_state, cell_state = train.train_model(data, model, max_epochs = epochs, rate=learning_rate, weight_decay=weight_decay, shuffle=shuffle)

    open_list = []
    close_list=[]
    predict_list=[]
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
        price_scaler = MinMaxScaler()
        price_scaler.min_, price_scaler.scale_ = scaler.min_[-1], scaler.scale_[-1]
        prediction = price_scaler.inverse_transform([torch.flatten(y_predict).tolist()])
        if not predict_price:
            prediction = [prediction[0] + table["open"][train_size:].values]
        actual = [table["closeadj"][train_size:].values]
        open_price = table["open"][train_size:].values
        prev_day_close = [table["closeadj"][train_size-1:-1].values]

        open_list.append(open_price)
        close_list.append(actual[0])
        predict_list.append(prediction[0])

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
        plt.legend(loc="lower right")
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        plt.savefig("{}_{}.png".format(ticker, count))
        #return MSE_Test
        # plt.show(block=False)
        # plt.pause(3)
        # plt.close()

    ##################
    # numberofstocks = len(predict_list)
    # endValues = []
    # benchmarkValues = []
    # temp = 100000
    # open_test = []
    # for j in range(numberofstocks):
    #     open_test.append(open_list[j][0])
    # shares = buyAndHold(open_test, 100000)  # [10,5]
    #
    # for i in range(len(predict_list[0])):
    #     y_pre = []
    #     close = []
    #     open = []
    #     bechmarkValue = 0
    #     for j in range(numberofstocks):
    #         y_pre.append(predict_list[j][i])
    #         close.append(close_list[j][i])
    #         open.append(open_list[j][i])
    #         bechmarkValue += shares[j] * close_list[j][i]
    #     end_vals = calculateReturn(y_pre, open, close, temp)
    #     temp = end_vals
    #     endValues.append(end_vals)
    #     benchmarkValues.append(bechmarkValue)
    #
    # fig, ax = plt.subplots()
    #
    # ax.plot(table["date"][train_size:], endValues, label='my strategy')
    # ax1 = ax
    # ax1.plot(table["date"][train_size:], benchmarkValues, label='benchmark')
    #
    # plt.xticks(rotation=30)
    # plt.legend(loc='upper left')
    # plt.savefig('return.png')

    # buy and hold


if __name__ == "__main__":
    main()
    # if not sys.warnoptions:
    #     warnings.simplefilter("ignore")
    # count = 0
    # min_args = None
    # min_MSE = float("inf")
    # for learning_rate in [0.01, 0.001]:
    #     for weight_decay in [0, 1e-4]:
    #         for hidden_size in [16, 32, 64]:
    #             for hidden_layer in [1, 2, 3]:
    #                 for lookback in [30, 60]:
    #                     for epochs in [100]:
    #                         for split in [0.8]:
    #                             for batch_size in [None, 128]:
    #                                 for stateless in [True, False]:
    #                                     for shuffle in [False]:
    #                                         for bidirection in [False]:
    #                                             for bias in [True, False]:
    #                                                 for predict_price in [True]:
    #                                                     count += 1
    #                                                     args = (count, learning_rate, weight_decay, hidden_size, hidden_layer, lookback, epochs, split, batch_size, stateless, shuffle, bidirection, bias, predict_price)
    #                                                     MSE_Test = main(args)
    #                                                     print(MSE_Test, count, args)
    #                                                     if MSE_Test < min_MSE:
    #                                                         min_MSE = MSE_Test
    #                                                         min_agrs = args
    # print("++++++++++++++++")
    # print(min_MSE, min_args)