import requests
from bs4 import BeautifulSoup
import pandas as pd
from os import path
import time
import datetime
import utilities
import SqliteManager


Utils = utilities.Utils
Sqm = SqliteManager.SqliteManager


def get_short_data_from_html(html):

    data = {}

    soup = BeautifulSoup(html, 'html.parser')

    short_table = soup.find('table')
    short_rows = short_table.find_all('tr')

    header = True
    for row in short_rows:

        if not header:

            row_data = row.find_all('td')
            date = ''
            short_interest = -1
            for rd in row_data:

                if rd['class'] == ['\\"col1\\"']:

                    # Replace the bad month names
                    full_date = rd.text.replace('.', '').replace('Sept', 'September').replace('Dec', 'December')
                    full_date = full_date.replace('Nov', 'November').replace('Oct', 'October').replace('Aug', 'August')
                    full_date = full_date.replace('Feb', 'February').replace('Jan', 'January')

                    date = datetime.datetime.strptime(full_date, '%B %d, %Y').strftime('%Y%m%d')
                elif rd['class'] == ['\\"col5\\"']:

                    short_interest_str = rd.text.split('\\n')[2].strip()
                    short_interest = Utils.short_num_str_to_int(short_interest_str)

            data[date] = short_interest

        header = False

    return data


def get_ycharts_short_interest_page(ticker):

    print('Getting {} short data'.format(ticker))

    username = 'sstben@gmail.com'
    password = '204436Brc!'

    # Start login session
    with requests.Session() as sess:

        login_cmd = 'https://ycharts.com/login'

        # Set CSRF cookie
        sess.get(login_cmd)
        if 'csrftoken' in sess.cookies:
            # Django 1.6 and up
            csrftoken = sess.cookies['csrftoken']
        else:
            # older versions
            csrftoken = sess.cookies['csrf']

        # Set up login info
        login_data = {'username': username, 'password': password, 'csrfmiddlewaretoken': csrftoken}
        login_header = {'referer': 'https://ycharts.com/'}

        sess.post(login_cmd, data=login_data, headers=login_header)  # Begin login session

        page_count = 1
        data_json = 'https://ycharts.com/companies/{}/short_interest.json?endDate={}%2F{}%2F{}&pageNum={}' \
                    '&startDate={}%2F{}%2F{}'.format(ticker, 12, 15, 2020, '{}', 1, 1, 1900)
        data_cmd = data_json.format(page_count)

        res = {}
        while True:
            try:
                data = sess.get(data_cmd)
                break
            except:
                time.sleep(5)
                continue

        if '404: Sorry' not in data.text:

            print('Retrieved page {}.'.format(page_count))

            # Get all pages of historical data
            repeating = False
            while not repeating:

                curr_html = data.text
                temp = get_short_data_from_html(curr_html)

                page_count += 1
                next_cmd = data_json.format(page_count)
                while True:
                    try:
                        next_page = sess.get(next_cmd)
                        break
                    except:
                        time.sleep(5)
                        continue

                # If just getting the same page, exit
                if curr_html != next_page.text:
                    print('Retrieved page {}.'.format(page_count))
                    data = next_page
                    res.update(temp)
                else:
                    res.update(temp)
                    repeating = True

        else:
            res = None

    return res


def get_ycharts_shorts_for_tickers(tickers):

    data = {}
    for ticker in tickers:
        shorts = get_ycharts_short_interest_page(ticker)
        if shorts:
            data[ticker] = shorts

    return data


def create_tables_from_ycharts(ycharts_table, ycharts_shorts):

    # First, create a set of all date entries in reverse sorted order
    date_set = set()
    for ticker, vals in ycharts_shorts.items():
        for date in vals.keys():
            date_set.add(date)
    date_set = list(date_set)
    date_set.sort(reverse=True)

    # Create query
    q = 'CREATE TABLE IF NOT EXISTS shorts (' \
        'Symbol TEXT PRIMARY KEY,'
    formatted_dates = ''
    for date in date_set:
        formatted_dates += '\'{}\','.format(date)
        q += '\'{}\' REAL,'.format(date)
    q = q[:-1] + ');'
    formatted_dates = formatted_dates[:-1]

    ycharts_table.execute_query(q)

    # Create query for each ticker row
    for ticker, vals in ycharts_shorts.items():

        q = 'INSERT INTO shorts (Symbol, {}) VALUES (\'{}\', '.format(formatted_dates, ticker)
        for date in date_set:
            if date in vals.keys():
                q += '{},'.format(vals[date])
            else:
                q += '{},'.format(-1)
        q = q[:-1] + ');'

        ycharts_table.execute_query(q)


def get_shorts_from_ycharts_sqlite(ycharts_table):

    df = pd.read_sql_query('SELECT * from shorts', ycharts_table.connection)
    df = df.set_index('Symbol')

    return df


def main():

    sqlite_table = '../data/Databases/ychart_shorts.sqlite'

    # Check if table already saved
    if not path.exists(sqlite_table):

        tickers = Utils.get_all_tickers('20201201')

        ycharts_shorts = get_ycharts_shorts_for_tickers(tickers)

        ycharts_table = Sqm(sqlite_table)

        create_tables_from_ycharts(ycharts_table, ycharts_shorts)

    else:

        ycharts_table = Sqm(sqlite_table)

    shorts_data = get_shorts_from_ycharts_sqlite(ycharts_table)
    Utils.write_dataframe_to_csv(shorts_data, '../data/ycharts_shorts.csv')


if __name__ == '__main__':
    main()