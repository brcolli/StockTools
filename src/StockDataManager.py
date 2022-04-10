from os import path
import yahoo_fin.stock_info as si
import utilities
import SqliteManager


Utils = utilities.Utils
Sqm = SqliteManager.SqliteManager


"""StockDataManager

Description:
Module for handling the stock data database. The stock data database stores all pertinent historical data for as many
tickers as we could find. Initially used AmiBroker's Yahoo Finance data calls, but also supports direct calls to the
Yahoo data using yahoo_fin. Has one table that contains a list of all the tickers, and then one table for each ticker,
containing open, high, low, close, adjusted close, and volume historical data as far back as possible and updated daily.

Authors: Benjamin Collins
Date: April 22, 2021 
"""


class StockDataManager:

    """Manager for the stock data database.
    """

    def __init__(self, ami_path='C:\Program Files\AmiBroker\AmiQuote\Download'):

        """Constructor method, opens and sets the database

        :param ami_path: Path to the AmiQuote download files;
                         defaults to C:\\Program Files\\AmiBroker\\AmiQuote\\Download
        :type ami_path: str
        """

        self.database_path = '../data/Databases/stock_data.sqlite'
        self.database = Sqm(self.database_path)
        self.amiquote_path = ami_path

    def add_ticker_data(self, symbol, data):

        """Adds ticker data to the database. Requires all historical data points for an entry:
        Date, High, Low, Close, Adjusted Close, and Volume. Can add multiple rows of data at once.

        :param symbol: Ticker symbol of data to be added
        :type symbol: str
        :param data: The data to add, can be a list of multiple rows
        :type data: list(tuple(object))
        """

        print('Adding data for {}.'.format(symbol))

        q = 'CREATE TABLE IF NOT EXISTS [{}] (' \
            'Date TEXT PRIMARY KEY,' \
            'Open REAL,' \
            'High REAL,' \
            'Low REAL,' \
            'Close REAL,' \
            'AdjClose REAL,' \
            'Volume REAL);'.format(symbol)
        self.database.execute_query(q)

        # Check if symbol exists in Symbol table
        if not self.database.check_if_key_exists('Symbols', 'Symbol', symbol):

            # Add symbol to Symbol table
            q = 'INSERT INTO Symbols (Symbol)' \
                'VALUES (\'{}\');'.format(symbol)
            self.database.execute_query(q)

        q = 'INSERT INTO [{}] (Date, Open, High, Low, Close, AdjClose, Volume) ' \
            'VALUES' \
            '(?, ?, ?, ?, ?, ?, ?);'.format(symbol)
        self.database.execute_many_query(q, data)

    def create_stock_stats(self):

        """Creates the base Symbols table and then creates all the ticker tables and updates to the most current.
        """

        self.database.execute_query('CREATE TABLE IF NOT EXISTS Symbols ('
                                    'id INTEGER PRIMARY KEY AUTOINCREMENT,'
                                    'Symbol TEXT);')

        self.update_stock_stats_all()

    def create_stock_stats_from_amiquote(self):

        """Goes through AmiQuote's download directory to generate data from each download file. Before calling,
        ensure AmiQuote data downloading has been called.
        """

        # Go through all files in amiquote folder and create table
        stock_files = Utils.find_file_pattern('*.aqh', self.amiquote_path)

        self.database.execute_query('CREATE TABLE IF NOT EXISTS Symbols ('
                                    'id INTEGER PRIMARY KEY AUTOINCREMENT,'
                                    'Symbol TEXT);')

        for sf in stock_files:

            with open(sf, 'r') as f:

                lines = f.readlines()
                symbol = ''
                headers = []

                # Find symbol name
                match = Utils.get_matches_from_regex(lines[0], '\$NAME\s(.*)\\n')
                if match:
                    symbol = match[0]

                # Skip key if already created
                if self.database.check_if_key_exists('Symbols', 'Symbol', symbol):
                    continue

                # Find header list
                match = Utils.get_matches_from_regex(lines[1], '^#\s(.*)')
                if match:
                    headers = match[0].split(',')

                if symbol == '' or headers == []:
                    print('Unable to parse file {}.'.format(sf))
                    continue

                # Get remainder of data and add to table, if parsed correctly
                line_data = lines[2:]
                data = []
                for i in line_data:
                    clean = i.replace('\n', '').replace('null', '-1').split(',')  # Replace newlines and nulls
                    data.append((Utils.reformat_time_str(clean[0]), clean[1], clean[2], clean[3], clean[4],
                                 clean[5], clean[6]))

                self.add_ticker_data(symbol, data)

    def update_stock_stats_with_amiquote(self, tickers=None):

        """Updates a list of ticker tables. Uses the AmiQuote download directory. Before updating, ensure AmiQuote
        has downloaded all the desired tickers.

        :param tickers: List of tickers to be updated; defaults to all tickers in the Symbols table
        :type tickers: list(str)
        """

        # If no list is given, assume to update all
        if not tickers:
            syms = self.database.execute_read_query('SELECT * FROM Symbols')
            tickers = [s[1] for s in syms]

        for ticker in tickers:

            sf = self.amiquote_path + '/{}.aqh'.format(ticker)
            if not path.exists(sf):
                print('Could not find amiquote data for {}.'.format(ticker))
                continue

            with open(sf, 'r') as f:

                lines = f.readlines()
                symbol = ''
                headers = []

                # Find symbol name
                match = Utils.get_matches_from_regex(lines[0], '\$NAME\s(.*)\\n')
                if match:
                    symbol = match[0]

                # Find header list
                match = Utils.get_matches_from_regex(lines[1], '^#\s(.*)')
                if match:
                    headers = match[0].split(',')

                if symbol == '' or headers == []:
                    print('Unable to parse file {}.'.format(sf))
                    continue

                # If key doesn't exist, update Symbols table and create new individual table
                if not self.database.check_if_key_exists('Symbols', 'Symbol', symbol):

                    # Add symbol to Symbol table
                    q = 'INSERT INTO Symbols (Symbol)' \
                        'VALUES (\'{}\');'.format(symbol)
                    self.database.execute_query(q)

                    # Create new table
                    q = 'CREATE TABLE IF NOT EXISTS [{}] (' \
                        'Date TEXT PRIMARY KEY,' \
                        'Open REAL,' \
                        'High REAL,' \
                        'Low REAL,' \
                        'Close REAL,' \
                        'AdjClose REAL,' \
                        'Volume REAL);'.format(symbol)
                    self.database.execute_query(q)

                print('Updating data for {}.'.format(symbol))

                # Get remainder of data and add to table, if parsed correctly
                line_data = lines[2:]
                data = []
                last_updated_date = Utils.time_str_to_datetime(self.database.get_last_row('[{}]'.format(symbol),
                                                                                          'Date')[0])

                for i in line_data:

                    clean = i.replace('\n', '').replace('null', '-1').split(',')  # Replace newlines and nulls

                    # If the date is older than or equal to the oldest date in the datebase, skip
                    if Utils.time_str_to_datetime(clean[0]) <= last_updated_date:
                        continue

                    data.append((Utils.reformat_time_str(clean[0]), clean[1], clean[2], clean[3], clean[4],
                                 clean[5], clean[6]))

                q = 'INSERT INTO [{}] (Date, Open, High, Low, Close, AdjClose, Volume) ' \
                    'VALUES' \
                    '(?, ?, ?, ?, ?, ?, ?);'.format(symbol)
                self.database.execute_many_query(q, data)

    def update_stock_stats_all(self):

        """Updates all stock data in the database. Uses yahoo_fin library, not AmiQuote. This is the suggested method
        for updating the database.
        """

        tickers = Utils.get_all_tickers()

        start_day = Utils.time_str_to_datetime('19700102')
        end_day = Utils.get_last_trading_day()

        # For each ticker, get data from yahoo_fin and send a vectorized command to update the ticker table
        for ticker in tickers:

            try:
                ps = si.get_data(ticker, start_date=start_day, end_date=end_day)
            except Exception as e:
                print('Bad data for {} due to {}. Skipping.'.format(ticker, e))
                continue

            if ps.empty:
                continue

            # Remove ticker column and convert index to datetime string
            ps = ps.drop(['ticker'], axis=1)
            ps.index = ps.index.map(lambda d: Utils.datetime_to_time_str(d.date()))

            # Convert dataframe to list of tuples
            data = list(ps.itertuples(name=None))

            self.add_ticker_data(ticker, data)

    def write_stock_data_dataframe_to_sqlite(self, df):

        for row, col in df.iterrows():
            self.add_ticker_data(row, [(col['Date'], col['Open'], col['High'], col['Low'], col['Unadjusted Close'],
                                       col['Close'], col['Total Volume'])])


def main():

    sdm = StockDataManager()

    if not sdm.database.database_empty():
        sdm.create_stock_stats()


if __name__ == '__main__':
    main()
