import numpy as np
def calculateReturn(y_predict, open, adjclose, start_val=100000, commission=0, impact=0):
    '''
    :param y_predict: list
    :param open: list
    :param close: list
    :param start_val:
    :param commission:
    :param impact:
    :param numberOfStocks:
    :return:
    '''
    numberOfStocks = len(y_predict)
    each_start = start_val/numberOfStocks
    shares = np.zeros(numberOfStocks)
    vals = np.ones(numberOfStocks) * each_start
    for i in range(len(y_predict)):
        if y_predict[i]>open[i]: #predicted close price is larger than open price, buy at close and sell in the end of the day
            shares[i] = each_start/open[i]
            vals[i] = shares[i]*adjclose[i] - commission
    end_vals = sum(vals)
    return end_vals



def buyAndHold(open,start_val=100000, commission=0, impact=0):
    """
    :param open: open price on the first test date []
    :param start_val:
    :param commission:
    :param impact:
    :return:shares
    """

    numberOfStocks = len(open)
    each_start = start_val / numberOfStocks
    shares = np.zeros(numberOfStocks)
    
    for i in range(len(open)):
        shares[i] = each_start/open[i]
        
    return shares
    





