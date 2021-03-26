import importlib
import pandas as pd
from yahoo_earnings_calendar import YahooEarningsCalendar
import yfinance as yf
from os import path
import sys

try:
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename
except ImportError:
    pass


TCM = importlib.import_module('TdaClientManager').TdaClientManager
Utils = importlib.import_module('utilities').Utils


class EarningsManager:

    def __init__(self, criteria):

        self.yec = YahooEarningsCalendar()
        self.criteria = criteria

    def get_earnings(self, date_range):

        filename = '../data/Earnings/' + Utils.get_proper_date_format(date_range[0]) + '_' + \
                   Utils.get_proper_date_format(date_range[1]) + '_earnings.csv'

        # If datafile already exists, load from that instead
        if path.exists(filename):
            data = Utils.load_csv_to_dataframe(filename)
        else:
            # File doesn't exist, write to it

            try:
                data_dict = self.yec.earnings_between(date_range[0], date_range[1])
            except:
                print('Yahoo Finance Earnings Calender is down. Try again later.')
                return pd.DataFrame()

            data = pd.DataFrame.from_dict(data_dict)

            # Remove unused columns
            unused = ['startdatetimetype', 'timeZoneShortName', 'gmtOffsetMilliSeconds', 'quoteType']
            data.drop(unused, axis=1, inplace=True)

            # Delete any duplicate rows
            data.drop_duplicates(subset='ticker', keep='first', inplace=True)

            # Add more columns of interest
            ticker_data = {'volume': {}, 'marketCap': {}, 'lastClose': {}}

            # Go through each ticker to find ones that don't meet certain criteria
            tickers = data['ticker'].tolist()

            for ticker in tickers:

                try:
                    curr_info = yf.Ticker(ticker).info
                except KeyError:
                    print('Bad data for {}. Skipping.'.format(ticker))
                    data = data[data.ticker != ticker]
                    continue

                curr_info_keys = curr_info.keys()

                if 'volume' in curr_info_keys and 'marketCap' in curr_info_keys and\
                   'previousClose' in curr_info_keys and 'currency' in curr_info_keys:

                    # Attempt to get ticker data
                    vl = curr_info['volume']
                    mc = curr_info['marketCap']
                    lc = curr_info['previousClose']

                else:
                    print('Bad data on ticker ' + ticker + '. Deleting from results.')
                    data = data[data.ticker != ticker]
                    continue

                # Check if passes criteria
                if self.criteria(vl, mc, lc):

                    ticker_data['volume'][vl] = ticker
                    ticker_data['marketCap'][mc] = ticker
                    ticker_data['lastClose'][lc] = ticker
                else:
                    print('Ticker ' + ticker + ' did not meet the criteria. Deleting from results.')
                    data = data[data.ticker != ticker]

            # Add new columns that passed criteria to dataframe
            data['volume'] = ticker_data['volume']
            data['marketCap'] = ticker_data['marketCap']
            data['lastClose'] = ticker_data['lastClose']

            data.reset_index(drop=True, inplace=True)  # Reset indices after removing

        # Rename column from ticker to symbol
        data = data.rename(columns={'ticker': 'Symbol'})

        Utils.write_dataframe_to_csv(data, filename)

        return data

    def get_earnings_in_range(self, date_range):
        return self.get_earnings(date_range)

    def get_next_week_earnings(self):
        week_range = Utils.get_following_week_range()
        return self.get_earnings(week_range)

    @staticmethod
    def import_earnings(filename=''):

        if filename == '' and 'tkinter' in sys.modules:
            Tk().withdraw()
            filename = askopenfilename()

        return pd.read_csv(filename)

    @staticmethod
    def find_options_from_earnings(es):

        tcm = TCM()
        opts = tcm.find_options(es['Symbol'].to_list())

        # Write to file
        for sym, sym_opts in opts.items():
            print('Writing earnings options for {}.'.format(sym))
            Utils.write_dataframe_to_csv(sym_opts, '../data/Earnings/Options/{}_earnings_options.csv'.format(sym))

        return opts


def main(ymd1='', ymd2='', min_vl=1E6, min_mc=3E8, min_lc=10):

    # min_vl = 1E6  # Minimum volume
    # min_mc = 3E8  # Minimum market cap
    # min_lc = 10   # Minimum last closed value

    def criteria(vl, mc, lc): return vl >= min_vl and mc >= min_mc and lc >= min_lc

    em = EarningsManager(criteria)

    if ymd1 == '':
        print('Getting next week\'s earnings.')
        es = em.get_next_week_earnings()
    else:
        if ymd2 == '':
            print('Please pass an end date')
            return

        es = em.get_earnings_in_range((ymd1, ymd2))

    if not es.empty:
        em.find_options_from_earnings(es)
    else:
        try:
            df = em.import_earnings()
        except FileNotFoundError:
            return

        em.find_options_from_earnings(df)


if __name__ == '__main__':
    main('20210405', '20210409')

