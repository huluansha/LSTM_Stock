import quandl
import numpy as np
import torch

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

quandl.ApiConfig.api_key="vrJkt5c3B5qJRExuYYPg"

class data_loader:
    def __init__(self, seq_len = 20, batch_size = 4):
        self.seq_len = seq_len
        self.batch_size = batch_size

    def load_data(self, tickers = ['AAPL'], start = '2019-01-01', end = '2020-01-01', numbers = 0):

        for ticker in tickers:
            table = quandl.get_table('SHARADAR/SEP',
                                     qopts={"columns":['ticker' ,'date','open','high','low','close','volume','closeadj']},
                                     date={'gte': start, 'lte': end},
                                     ticker=ticker)
            if table.shape[0] < numbers:
                raise("Not Enough Data for tick {}".format(ticker))
            else:
                feature_sequence = table[['open','high','low','volume','closeadj']].values
                training_size = int(table.shape[0] * 0.8)

                feature_train = feature_sequence[0:training_size]
                feature_test = feature_sequence[training_size:-1]

                scaler = MinMaxScaler(feature_range=(0,1))
                scaler.fit(feature_train)

                feature_train = scaler.transform(feature_train)
                feature_test = scaler.transform(feature_test)

                train_x, train_y = [], []
                test_x, test_y = [], []
                for i in range(feature_train.shape[0] - self.seq_len):
                    sub_feature = feature_train[i:i+self.seq_len]
                    sub_price_diff = feature_train[i+self.seq_len, 4]
                    train_x.append(sub_feature)
                    train_y.append(sub_price_diff)

                for i in range(feature_test.shape[0] - self.seq_len):
                    sub_feature = feature_train[i:i+self.seq_len]
                    sub_price_diff = feature_train[i+self.seq_len, 4]
                    test_x.append(sub_feature)
                    test_y.append(sub_price_diff)

                train_x, train_y = torch.FloatTensor(train_x[0:(len(train_x)//self.batch_size * self.batch_size)]), \
                                   torch.FloatTensor(train_y[0:(len(train_x)//self.batch_size * self.batch_size)])

                test_x, test_y = torch.FloatTensor(test_x[0:(len(test_x)//self.batch_size * self.batch_size)]), \
                                   torch.FloatTensor(test_y[0:(len(test_y)//self.batch_size * self.batch_size)])

                train_x = train_x.view(train_x.shape[0]//self.batch_size, self.batch_size, self.seq_len, 5)
                train_y = train_y.view(train_y.shape[0] // self.batch_size, self.batch_size, 1)
                test_x = test_x.view(test_x.shape[0]//self.batch_size, self.batch_size, self.seq_len, 5)
                test_y = test_y.view(test_y.shape[0] // self.batch_size, self.batch_size, 1)


        return train_x, train_y, test_x, test_y, scaler


if __name__ == "__main__":
    loader = data_loader()
    train_x, train_y, test_x, test_y, scaler = loader.load_data()


