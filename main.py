import train
from LSTM import LSTM
from data_loader import data_loader
import matplotlib.pyplot as plt

def main():
    model = LSTM()
    loader = data_loader()

    hidden_state, cell_state = train.train_model(loader, model, max_epochs = 10, seq_length = loader.seq_len)
    #hidden_state, cell_state = model.init_state(1)
    test_data = loader.load_data(tickers = ['AAPL'], start = '2020-01-02', end = '2021-07-01', numbers = 0)

    test_diff = []
    x_axis_val = range(len(test_data))
    for batch, (x, y) in test_data:
        y_predict, _ = model(x, (hidden_state, cell_state))
        test_diff.append([y.tolist()[0], y_predict.tolist()[0]])


    # plt.plot(x_axis_val, [x[0] for x in test_diff], label = 'actual')
    plt.plot(x_axis_val, [x[1] for x in test_diff], label = 'predict')
    plt.savefig("aapl.png")
    plt.legend()
    plt.show()
    plt.close()



if __name__ == "__main__":
    main()