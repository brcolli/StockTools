import importlib
from os import path


Utils = importlib.import_module('utilities').Utils
Sqm = importlib.import_module('SqliteManager').SqliteManager
TCM = importlib.import_module('TdaClientManager').TdaClientManager


class StockDataManager:

    def __init__(self, ami_path='C:\Program Files\AmiBroker\AmiQuote\Download'):

        self.database_path = '../data/Databases/stock_data.sqlite'
        self.database = Sqm(self.database_path)
        self.amiquote_path = ami_path

    def add_ticker_data(self, symbol, data):

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

    def create_stock_stats_from_amiquote(self):

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

    def update_stock_stats(self, tickers=None):

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

        tickers = Utils.get_all_tickers()

        start_day = Utils.time_str_to_datetime('19700102')
        end_day = Utils.time_str_to_datetime('20210410')

        tcm = TCM()

        for ticker in tickers:

            ps = tcm.get_past_history([ticker], start_day, end_day)
            if not ps:
                continue
            val = ps[ticker]

            data = []
            for i in range(len(val)):
                day = val[i]

                if i == 0:
                    data.append((Utils.epoch_to_time_str(day['datetime']), day['open'], day['high'], day['low'],
                                 day['close'], 0, day['volume']))
                else:
                    data.append((Utils.epoch_to_time_str(day['datetime']), day['open'], day['high'], day['low'],
                                 day['close'], val[i-1]['close'], day['volume']))

            self.add_ticker_data(ticker, data)


def main():

    sdm = StockDataManager('C:\Program Files\AmiBroker\AmiQuote\Download')

    if not path.exists(sdm.database_path):
        sdm.create_stock_stats_from_amiquote()

    #sdm.update_stock_stats()
    sdm.update_stock_stats_all()


if __name__ == '__main__':
    main()
