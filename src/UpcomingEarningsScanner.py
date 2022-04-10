import pandas as pd
import yahoo_fin.stock_info as si
from os import path
import sys
import TdaClientManager
import utilities

try:
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename
except ImportError:
    Tk = None
    askopenfilename = None
    pass


TCM = TdaClientManager.TdaClientManager
Utils = utilities.Utils


class EarningsManager:

    def __init__(self, criteria):

        self.criteria = criteria

    def get_earnings(self, date_range):

        filename = '../data/Earnings/' + Utils.get_proper_date_format(date_range[0]) + '_' + \
                   Utils.get_proper_date_format(date_range[1]) + '_earnings.csv'

        # If datafile already exists, load from that instead
        if path.exists(filename) and False:
            data = Utils.load_csv_to_dataframe(filename)
        else:

            # File doesn't exist, write to it
            try:
                data_dict = si.get_earnings_in_date_range(date_range[0], date_range[1])
            except:
                print('Yahoo Finance Earnings Calender is down. Try again later.')
                return pd.DataFrame()

            data = pd.DataFrame.from_dict(data_dict)

            if data.empty:
                print('Failed to get earnings data, please try again.')
                return data

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
                    curr_info = si.get_quote_data(ticker)
                except Exception as e:
                    print('Bad data for {} due to {}. Skipping.'.format(ticker, e))
                    data = data[data.ticker != ticker]
                    continue

                curr_info_keys = curr_info.keys()

                if 'regularMarketVolume' in curr_info_keys and 'marketCap' in curr_info_keys and\
                   'regularMarketPreviousClose' in curr_info_keys and 'currency' in curr_info_keys:

                    # Attempt to get ticker data
                    vl = curr_info['regularMarketVolume']
                    mc = curr_info['marketCap']
                    lc = curr_info['regularMarketPreviousClose']

                else:
                    print('Bad data on ticker ' + ticker + '. Deleting from results.')
                    data = data[data.ticker != ticker]
                    continue

                # Check if passes criteria
                if self.criteria(vl, mc, lc):

                    ticker_data['volume'][ticker] = vl
                    ticker_data['marketCap'][ticker] = mc
                    ticker_data['lastClose'][ticker] = lc
                else:
                    print('Ticker ' + ticker + ' did not meet the criteria. Deleting from results.')
                    data = data[data.ticker != ticker]

            # Add new columns that passed criteria to dataframe
            data['volume'] = ticker_data['volume'].values()
            data['marketCap'] = ticker_data['marketCap'].values()
            data['lastClose'] = ticker_data['lastClose'].values()

            data.reset_index(drop=True, inplace=True)  # Reset indices after removing

        # Rename column from ticker to symbol
        data = data.rename(columns={'ticker': 'Symbol'})

        Utils.write_dataframe_to_csv(data, filename)

        # Upload to drive
        sub_dir = 'Upcoming Earnings/' + '/'.join(filename.split('/')[3:-1])
        Utils.upload_files_to_gdrive([filename], sub_dir)

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
    main()
