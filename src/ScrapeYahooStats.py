import requests
import importlib
import json
from os import path
import time
import yfinance as yf


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


def add_sector_column_to_fundamentals(stats_table, tickers):

    # Collect sector data from yfinance
    data = []
    for ticker in tickers:

        print('Getting sector data for {}'.format(ticker))
        t = yf.Ticker(ticker)

        try:
            sector = t.info['sector']
        except Exception as e:

            if e == KeyError:
                sector = t.info['quoteType']
            else:
                sector = 'Unknown'

        data.append((sector, ticker))

    stats_table.add_new_column_with_data('fundamentals', 'Symbol', 'sector', 'TEXT', data)


def main():

    yahoo_table = '../data/Databases/yahoo_stats.sqlite'
    tickers = Utils.get_all_tickers('20201201')

    if not path.exists(yahoo_table):

        yahoo_stats = get_yahoo_stats_for_tickers(tickers)

        # Set up database
        stats_table = Sqm(yahoo_table)

        create_table_from_yahoo_stats(stats_table, yahoo_stats)

    else:
        stats_table = Sqm(yahoo_table)

    add_sector_column_to_fundamentals(stats_table, tickers)


if __name__ == '__main__':
    main()
