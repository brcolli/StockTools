from os import path
import utilities
import TdaClientManager
import SqliteManager


Utils = utilities.Utils
TCM = TdaClientManager.TdaClientManager
Sqm = SqliteManager.SqliteManager


def get_historical_closing(tcm, tickers, lookback):

    date_range = Utils.get_bd_range_from_count(lookback)
    rsi_history = tcm.get_past_history(tickers, Utils.time_str_to_datetime(date_range[-1]),
                                       Utils.time_str_to_datetime(date_range[0]))

    # Validate that all tickers have all dates
    for key, val in rsi_history.items():

        date_diff = abs(len(val) - len(date_range))
        if date_diff != 0:

            # If there's a difference, attempt to get excess missing days
            new_date_range = Utils.get_bd_range_from_count(lookback + date_diff)

            missing_history = tcm.get_past_history([key], Utils.time_str_to_datetime(new_date_range[-1]),
                                                   Utils.time_str_to_datetime(new_date_range[0]))

            for key2, val2 in missing_history.items():

                date_diff = abs(len(val2) - len(date_range))
                if date_diff != 0:
                    # If still don't have enough, assume there's not enough available data and append to fill
                    print('Warning, could not find {} data points for {}.'.format(lookback, key2))
                    val += [{'close': -1}] * date_diff

    return rsi_history


def create_table_from_historical(rsi_table, data):

    rsi_table.execute_query('CREATE TABLE IF NOT EXISTS Symbols ('
                            'id INTEGER PRIMARY KEY AUTOINCREMENT,'
                            'Symbol TEXT);')

    for key, val in data.items():

        q = 'INSERT INTO Symbols (Symbol)' \
            'VALUES' \
            '(\'{}\');'.format(key)
        rsi_table.execute_query(q)

        # Create individual tickers table
        q = 'CREATE TABLE IF NOT EXISTS {} (' \
            'id INTEGER PRIMARY KEY AUTOINCREMENT,' \
            'Close REAL);'.format(key)
        rsi_table.execute_query(q)

        q = 'INSERT INTO {} (Close)' \
            'VALUES '.format(key)

        for entry in val:
            q += '({}),'.format(entry['close'])
        q = q[:-1] + ';'

        rsi_table.execute_query(q)


def get_historical_closing_from_sqlite(rsi_table):

    tickers = rsi_table.execute_read_query('SELECT * from Symbols')
    data = {}

    for ticker_tuple in tickers:

        q = 'SELECT * from {}'.format(ticker_tuple[1])
        temp = rsi_table.execute_read_query(q)
        data[ticker_tuple[1]] = [data_tuple[1] for data_tuple in temp]

    return data


def main(momentum_period, trend_period, relation_period):

    if momentum_period > relation_period or trend_period > relation_period:
        print('Relational trend period must be the largest of the trend values')
        return

    rsi_table = '../data/Databases/connors_rsi.sqlite'
    tcm = TCM()

    # Check if table already saved
    if not path.exists(rsi_table):

        tickers = Utils.get_all_tickers('20201201')[:10]

        rsi_history = get_historical_closing(tcm, tickers, relation_period)

        rsi_table = Sqm(rsi_table)

        create_table_from_historical(rsi_table, rsi_history)

    else:
        rsi_table = Sqm(rsi_table)

    history_vals = get_historical_closing_from_sqlite(rsi_table)


if __name__ == '__main__':
    main(3, 2, 100)
