import requests
import importlib
import json
import time


Utils = importlib.import_module('utilities').Utils


def get_nasdaq_trade_order(ticker):

    print('Getting {} data'.format(ticker))

    data_cmd = 'https://www.nasdaq.com/market-activity/stocks/{}/latest-real-time-trades'.format(ticker)

    while True:
        try:
            data = requests.get(data_cmd)
            break
        except:
            time.sleep(5)
            continue

    if '404 Not Found' not in data.text:
        print(data.text)
        res = data.text
    else:
        res = None

    return res


def get_nasdaq_trade_orders(tickers):

    res = {}
    for ticker in tickers:
        res[ticker] = get_nasdaq_trade_order(ticker)

    return res


def main():

    tickers = ['AMC']
    data = get_nasdaq_trade_orders(tickers)


if __name__ == '__main__':
    main()