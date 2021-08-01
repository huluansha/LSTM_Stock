import quandl
import numpy as np
import torch

from sklearn.preprocessing import MinMaxScaler,StandardScaler

quandl.ApiConfig.api_key="vrJkt5c3B5qJRExuYYPg"

class data_loader:
    def __init__(self, seq_len, select_features):
        self.seq_len = seq_len
        self.select_features = select_features

    def load_data(self, tickers, start, end, split, predict_price, min_days = 0):
        data_holder = []
        for ticker in tickers:
            table = quandl.get_table('SHARADAR/SEP',
                                     qopts={"columns":['ticker' ,'date','open','high','low','close','volume','closeadj']},
                                     date={'gte': start, 'lte': end},
                                     ticker=ticker)

            table = table.reindex(index=table.index[::-1]) # reverse, table[0] is the earliest day

            if table.shape[0] < min_days:
                raise("Not Enough Data for tick {}".format(ticker))
            else:
                feature_sequence = table[self.select_features].values
                training_size = int(table.shape[0] * split)
                if training_size <= self.seq_len:
                    print("not enough data")
                    continue
                feature_train = feature_sequence[0:training_size]

                # scaler = MinMaxScaler(feature_range=(-1,1))
                scaler = MinMaxScaler(feature_range=(-1,1))
                scaler.fit(feature_train)

                feature_sequence = scaler.transform(feature_sequence)


                train_x, train_y = [], []
                test_x, test_y = [], []
                for i in range(self.seq_len, training_size):
                    x = feature_sequence[i - self.seq_len:i]
                    if predict_price:
                        y = feature_sequence[i, -1]
                    else: # predict diff
                        y = feature_sequence[i, -1] -  feature_sequence[i - 1, -1]
                    train_x.append(x)
                    train_y.append(y)

                for i in range(training_size, len(feature_sequence)):
                    x = feature_sequence[i - self.seq_len:i]
                    if predict_price:
                        y = feature_sequence[i, -1]
                    else: # predict diff
                        y = feature_sequence[i, -1] -  feature_sequence[i - 1, -1]
                    test_x.append(x)
                    test_y.append(y)

                # train_x, train_y = torch.FloatTensor(train_x[0:(len(train_x)//self.batch_size * self.batch_size)]), \
                #                    torch.FloatTensor(train_y[0:(len(train_x)//self.batch_size * self.batch_size)])
                #
                # test_x, test_y = torch.FloatTensor(test_x[0:(len(test_x)//self.batch_size * self.batch_size)]), \
                #                    torch.FloatTensor(test_y[0:(len(test_y)//self.batch_size * self.batch_size)])
                #
                # train_x = train_x.view(train_x.shape[0]//self.batch_size, self.batch_size, self.seq_len, 5)
                # train_y = train_y.view(train_y.shape[0] // self.batch_size, self.batch_size, 1)
                # test_x = test_x.view(test_x.shape[0]//self.batch_size, self.batch_size, self.seq_len, 5)
                # test_y = test_y.view(test_y.shape[0] // self.batch_size, self.batch_size, 1)

                train_x, train_y = torch.FloatTensor(train_x), \
                                   torch.FloatTensor(train_y)

                test_x, test_y = torch.FloatTensor(test_x), \
                                 torch.FloatTensor(test_y)

                train_x = train_x.view(train_x.shape[0], train_x.shape[1], len(self.select_features))
                train_y = train_y.view(train_y.shape[0], 1)
                test_x = test_x.view(test_x.shape[0], test_x.shape[1], len(self.select_features))
                test_y = test_y.view(test_y.shape[0], 1)
                data_holder.append((train_x, train_y, test_x, test_y, scaler, table))

        return data_holder





