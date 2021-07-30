import quandl
import numpy as np
import torch

quandl.ApiConfig.api_key="vrJkt5c3B5qJRExuYYPg"

class data_loader:
    def __init__(self, seq_len = 20):
        self.seq_len = seq_len

    def load_data(self, tickers = ['AAPL'], start = '2019-01-01', end = '2020-01-01', numbers = 0):

        # Build Training Set
        data = []
        for ticker in tickers:
            table = quandl.get_table('SHARADAR/SEP',
                                     qopts={"columns":['ticker' ,'date','open','high','low','close','volume','closeadj']},
                                     date={'gte': start, 'lte': end},
                                     ticker=ticker)
            if table.shape[0] < numbers:
                raise("Not Enough Data for tick {}".format(ticker))
            else:
                feature_sequence = table[['open','high','low','volume','closeadj']].values
                for i in range(table.shape[0] - self.seq_len):
                    batch_data = torch.FloatTensor([feature_sequence[i:i+self.seq_len][:]])
                    batch_return = torch.FloatTensor([feature_sequence[i + self.seq_len][4] - feature_sequence[i + self.seq_len][0]])
                    data_return = (batch_data, batch_return)
                    data.append((i, data_return))


        return data





