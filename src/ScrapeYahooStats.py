import requests
import importlib
import json
from os import path
import time


Utils = importlib.import_module('utilities').Utils
Sqm = importlib.import_module('SqliteManager').SqliteManager


def get_yahoo_stat_for_ticker(ticker):

    print('Getting {} data'.format(ticker))

    data_cmd = 'https://query1.finance.yahoo.com/v10/finance/quoteSummary/{}?formatted=true&crumb=WfVISMhWzBE&lang=' \
               'en-US&region=US&modules=defaultKeyStatistics%2CfinancialData%2CcalendarEvents&corsDomain=finance.' \
               'yahoo.com'.format(ticker)

    while True:
        try:
            data = requests.get(data_cmd)
            break
        except:
            time.sleep(5)
            continue

    if '404 Not Found' not in data.text:
        parsed = json.loads(data.text)
        res = parsed['quoteSummary']
    else:
        res = {'result': None}

    return res


def get_yahoo_stats_for_tickers(tickers):

    res = {}
    for ticker in tickers:
        stats = get_yahoo_stat_for_ticker(ticker)
        if stats['result'] is not None:
            res[ticker] = stats['result'][0]

    return res


def create_table_from_yahoo_stats(stats_table, yahoo_stats):

    stats_table.execute_query('CREATE TABLE IF NOT EXISTS fundamentals ('
                              'Symbol TEXT PRIMARY KEY,'
                              'Floats REAL);')

    for key, val in yahoo_stats.items():

        # Ignore any tickers with no float data
        try:
            floats = val['defaultKeyStatistics']['floatShares']['raw']
        except KeyError:
            continue

        q = 'INSERT INTO fundamentals (Symbol, Floats)' \
            'VALUES' \
            '(\'{}\', {});'.format(key, floats)
        stats_table.execute_query(q)


def main():

    yahoo_table = '../data/yahoo_stats.sqlite'

    if not path.exists(yahoo_table):

        tickers = Utils.get_all_tickers('20201201')

        yahoo_stats = get_yahoo_stats_for_tickers(tickers)

        # Set up database
        stats_table = Sqm(yahoo_table)

        create_table_from_yahoo_stats(stats_table, yahoo_stats)


if __name__ == '__main__':
    main()
