import importlib
import pandas as pd
from yahoo_earnings_calendar import YahooEarningsCalendar
import yfinance as yf
from os import path


Utils = importlib.import_module('utilities').Utils


class EarningsManager:

    def __init__(self, criteria):

        self.yec = YahooEarningsCalendar()
        self.criteria = criteria

    def get_earnings(self, date_range):

        filename = '../data/' + Utils.get_proper_date_format(date_range[0]) + '_' + \
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
                return pd.DataFrame.empty

            data = pd.DataFrame.from_dict(data_dict)

            # Remove unused columns
            unused = ['startdatetimetype', 'timeZoneShortName', 'gmtOffsetMilliSeconds', 'quoteType']
            data.drop(unused, axis=1, inplace=True)

            # Add more columns of interest
            ticker_data = {'volume': {}, 'marketCap': {}, 'lastClose': {}}

            # Go through each ticker to find ones that don't meet certain criteria
            tickers = data['ticker'].tolist()

            for ticker in tickers:

                curr_share = yf.Ticker(ticker)
                try:

                    # Attempt to get ticker data
                    curr_info = curr_share.info
                    vl = curr_info['volume']
                    mc = curr_info['marketCap']
                    lc = curr_info['previousClose']
                    cr = curr_info['currency']

                except:
                    print('Bad data on ticker ' + ticker + '. Deleting from results.')
                    data = data[data.ticker != ticker]
                    continue

                # Check if passes criteria
                if self.criteria(vl, mc, lc, cr):
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

            data.reset_index(drop=True, inplace=True) # Reset indices after removing

        Utils.write_dataframe_to_csv(data, filename)

        return data

    def get_earnings_in_range(self, date_range):
        return self.get_earnings(date_range)

    def get_next_week_earnings(self):

        week_range = Utils.get_following_week_range()
        data = self.get_earnings(week_range)

        return data


def main(min_vl, min_mc, min_lc):

    # min_vl = 1E6  # Minimum volume
    # min_mc = 3E8  # Minimum market cap
    # min_lc = 1E1  # Minimum last closed value

    def criteria(vl, mc, lc, cr): return vl >= min_vl and mc >= min_mc and lc >= min_lc

    em = EarningsManager(criteria)

    em.get_next_week_earnings()

    d1 = Utils.time_str_to_datetime('2020/09/13')
    d2 = Utils.time_str_to_datetime('2020/09/19')
    #em.get_earnings_in_range((d1, d2))


if __name__ == '__main__':
    main(min_vl=1E6,min_mc=3E8,min_lc=1E1)
