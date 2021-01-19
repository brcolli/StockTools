import importlib
import string
import pandas as pd
from os import path
from os import listdir


Utils = importlib.import_module('utilities').Utils
Sqm = importlib.import_module('SqliteManager').SqliteManager


class DatabaseManager:

    def __init__(self):

        self.fundamentals = Sqm('../data/yahoo_stats.sqlite')
        self.nasdaq_shorts = Sqm('../data/nasdaq_shorts.sqlite')
        self.shorts = Sqm('../data/ychart_shorts.sqlite')

    def get_ticker_data(self, tickers, reqs=''):

        # Append as many AND statements as needed
        q = 'SELECT * from shorts WHERE Symbol='

        for ticker in tickers:
            q += '\"{}\" OR Symbol='.format(ticker)
        q = q[:-11]  # Remove the extra ending

        data = self.shorts.execute_read_query(q)
        headers = self.shorts.get_column_names('shorts')
        headers = headers[1:]  # Remove the symbol column

        if data:

            # Add to a dictionary
            ret = {}
            for row in data:

                temp = {}
                for i in range(len(headers)):
                    temp[headers[i]] = row[i+1]
                ret[row[0]] = temp

            return pd.DataFrame(ret).transpose()
        else:
            return pd.DataFrame()

    @staticmethod
    def write_ticker_data(ticker_data, suffix):

        out_path = '../data/analysis/'
        for index, row in ticker_data.iterrows():
            Utils.write_dataframe_to_csv(row, out_path + index + suffix + '.csv')

    @staticmethod
    def write_tickers_composite(ticker_data, filename):

        # Transpose data for prettier output
        td = ticker_data.transpose()
        Utils.write_dataframe_to_csv(td, '../data/analysis/{}.csv'.format(filename))

    @staticmethod
    def get_data_from_amibroker():

        amibroker_data = 'C:/Program Files/AmiBroker/Data/'

        for letter in string.ascii_lowercase:

            curr_dir = amibroker_data + letter + '/'
            for filename in listdir(curr_dir):

                with open(curr_dir + filename, 'rb') as f:

                    data = f.read()
                    print(data.decode('ISO-8859-1'))

            break


def main():

    dbm = DatabaseManager()
    data = dbm.get_ticker_data(['AAPL', 'TSLA'])
    dbm.write_ticker_data(data, '_shorts')
    dbm.write_tickers_composite(data, 'Composite_shorts')


if __name__ == '__main__':
    main()
