import requests
import re
from bs4 import BeautifulSoup
import pandas as pd
from os import path
import time
import utilities
import SqliteManager


Utils = utilities.Utils
Sqm = SqliteManager.SqliteManager


# Scrape Nasdaq site for a single ticker value's short interest
def get_nasdaq_short_interest_page(ticker):

    print('Getting {} short data'.format(ticker))

    post_cmd = 'https://www.nasdaqtrader.com/RPCHandler.axd'
    headers = {'Content-type': 'application/json',
               'Referer': 'https://www.nasdaqtrader.com/Trader.aspx?id=ShortInterest',
               }

    data = {'method': 'BL_ShortInterest.SearchShortInterests', 'version': '1.1', 'params': '[\"{}\"]'.format(ticker)}
    while True:
        try:
            page = requests.post(post_cmd, json=data, headers=headers)
            break
        except:
            time.sleep(5)
            continue

    # Get results from page text
    r = re.search(r"\"result\":\"(.*)\"}", page.text)

    if not r:
        ret = 'Symbol Not Found'
    else:
        ret = r.group(1)

    return ret


# Parse html, looking for divShortInterestResults table
def get_short_data_from_html(html):

    if 'Symbol Not Found' in html:
        return pd.DataFrame()

    soup = BeautifulSoup(html, 'html.parser')

    short_table = soup.find('table')
    short_rows = short_table.find_all('tr')

    # Parse table data to dictionary
    header = []
    data = {}
    for row in short_rows:

        # If first row, consider as header
        if not header:
            row_data = row.find_all('th')
            for h in row_data:
                text = h.text.replace(' ', '')
                header.append(text)
                data[text] = []
        else:
            row_data = row.find_all('td')
            for i in range(len(row_data)):
                data[header[i]].append(row_data[i].text)

    return pd.DataFrame(data)


# For each ticker in a list, pull short info from Nasdaq and parse into dataframes
def get_nasdaq_short_for_tickers(tickers):

    data = {}
    for ticker in tickers:

        html = get_nasdaq_short_interest_page(ticker)
        df = get_short_data_from_html(html)

        # Construct dictionary of date to dataframe values
        if not df.empty:
            for key, val in df.iterrows():

                # Convert to YYYYMMDD
                date = val['SettlementDate']
                date = Utils.reformat_time_str(date)

                try:
                    curr_data = data[date]
                except KeyError:
                    curr_data = pd.DataFrame()

                # Dictionary for current row
                d = {'Symbol': ticker}
                for col in df.columns:
                    if col != 'SettlementDate':
                        try:
                            d[col] = float(val[col].replace(',', '').replace('(', '').replace(')', ''))
                        except ValueError:
                            d[col] = 0

                curr_data = curr_data.append(d, ignore_index=True)
                data[date] = curr_data

    # Set all indexing to use symbols
    for key, val in data.items():
        data[key] = val.set_index('Symbol')

    return data


# Store data from nasdaq into an sqlite database
def create_tables_from_nasdaq_shorts(shorts_table, data):

    # Create table of dates, which will be used as a reference for 2nd dimensional tables
    shorts_table.execute_query('CREATE TABLE IF NOT EXISTS dates ('
                               'id INTEGER PRIMARY KEY AUTOINCREMENT,'
                               'Date TEXT);')

    # For each date saved, add to entry in dates table and then create separate table
    for key, val in data.items():

        header = val.columns.tolist()

        # Insert data into the overall dates table
        q = 'INSERT INTO dates (Date)' \
            'VALUES' \
            '(\'{}\');'.format(key)
        shorts_table.execute_query(q)

        format_list = [key] + header[:4]

        # Create individual dates table
        q = 'CREATE TABLE IF NOT EXISTS Date{} (' \
            'Symbol TEXT PRIMARY KEY,' \
            '{} REAL,' \
            '{} REAL,' \
            '{} REAL,' \
            '{} REAL);'.format(*format_list)
        shorts_table.execute_query(q)

        # Insert data into individual dates table
        q = 'INSERT INTO Date{} (Symbol, {}, {}, {}, {}) ' \
            'VALUES '.format(*format_list)

        for index, row in val.iterrows():
            entry = [index] + [row[header[i]] for i in range(len(header))]
            q += '(\'{}\', {}, {}, {}, {}),'.format(*entry)
        q = q[:-1] + ';'

        shorts_table.execute_query(q)


# Get data from nasdaq short sqlite db and store in dataframes
def get_data_from_nasdaq_shorts_sqlite(shorts_table):

    dates = shorts_table.execute_read_query('SELECT * from dates')
    dfs = {}

    for date_tuple in dates:

        q = 'SELECT * from Date{}'.format(date_tuple[1])
        temp = pd.read_sql_query(q, shorts_table.connection)
        temp = temp.set_index('Symbol')
        dfs[date_tuple[1]] = temp

    return dfs


# Get data from all CSVs and store in dictionary to dataframe structure
def get_data_from_csvs(short_file_prefix):

    dfs = {}

    file_root = '../data_old/'
    csvs = Utils.find_file_pattern(short_file_prefix + '*', file_root)

    for csv in csvs:

        date = csv[len(file_root) + len(short_file_prefix):-4]
        data = pd.read_csv(csv)
        data = data.set_index('Symbol')
        dfs[date] = data

    return dfs


def calculate_short_difference(nasdata, csvdata):

    # Get and sort dates
    csvdates = list(csvdata.keys())
    csvdates.sort()

    short_pairs = {}  # A dictionary of tickers to a dictionary of dates to nasdaq, csv short interest pairs

    for date in csvdates:

        curr_csv = csvdata[date]
        # If on a NASDAQ date, sum up and find difference
        if date in nasdata:
            for key, val in curr_csv.iterrows():

                curr_nas = nasdata[date]
                if key in curr_nas.index:

                    nas_si = curr_nas.loc[key]['ShortInterest']

                    if val['ShortVolume'] != 0:
                        stats = {'nas': nas_si,
                                 'csv': val['ShortVolume'],
                                 'div': nas_si / val['ShortVolume'],
                                 'per': ((nas_si - val['ShortVolume']) / val['ShortVolume']) * 100}
                    else:
                        stats = {'nas': nas_si,
                                 'csv': val['ShortVolume'],
                                 'div': -1,
                                 'per': -1}

                    if key in short_pairs:
                        short_pairs[key][date] = stats
                    else:
                        short_pairs[key] = {date: stats}

    return short_pairs


def calculate_short_fudge(short_stats):

    fudges = {}
    for key, val in short_stats.items():

        date_count = 0
        fudge_sum = 0
        for row, col in val.items():
            fudge_sum += col['div']
            date_count += 1

        if key in fudges:
            fudges[key]['fudge'] = fudge_sum / date_count
        else:
            fudges[key] = {'fudge': fudge_sum / date_count}

    return fudges


def calculate_fudge_error(fudge_stats, short_stats):

    for key, val in short_stats.items():

        date_count = 0
        error_sum = 0
        for row, col in val.items():

            calculated = col['csv'] * fudge_stats[key]['fudge']
            error = abs((calculated - col['nas']) / col['nas']) * 100

            short_stats[key][row]['error'] = error

            error_sum += error
            date_count += 1

        fudge_stats[key]['error'] = error_sum / date_count

    return fudge_stats


def export_ticker_stats(tickers, short_stats):

    for ticker in tickers:
        ticker_stats = short_stats[ticker]
        ticker_df = pd.DataFrame(ticker_stats).transpose()
        ticker_df.index.name = 'Date'
        Utils.write_dataframe_to_csv(ticker_df, '../data/{}_short_stats.csv'.format(ticker))


def main():

    sqlite_table = '../data/Databases/nasdaq_shorts.sqlite'

    # Check if table already saved
    if not path.exists(sqlite_table):

        tickers = Utils.get_all_tickers('20201201')

        nasdaq_shorts = get_nasdaq_short_for_tickers(tickers)

        # Set up shorts database
        shorts_table = Sqm(sqlite_table)

        create_tables_from_nasdaq_shorts(shorts_table, nasdaq_shorts)

    else:
        shorts_table = Sqm(sqlite_table)

    nasdata = get_data_from_nasdaq_shorts_sqlite(shorts_table)
    csvdata = get_data_from_csvs('CNMSshvol')

    short_stats = calculate_short_difference(nasdata, csvdata)

    fudge_stats = calculate_short_fudge(short_stats)
    fudge_stats = calculate_fudge_error(fudge_stats, short_stats)

    tickers = ['AAPL', 'TSLA', 'AMD', 'FCEL', 'AAL', 'MRNA', 'SNDL', 'ONTX']
    export_ticker_stats(tickers, short_stats)

    fudge_df = pd.DataFrame(fudge_stats).transpose()
    Utils.write_dataframe_to_csv(fudge_df, '../data/fudge_factors.csv')

    return


if __name__ == '__main__':
    main()
