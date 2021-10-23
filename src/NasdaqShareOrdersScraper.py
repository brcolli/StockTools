import requests
import utilities
import json
import time
import datetime
import pandas as pd
from os import path
import os
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

Utils = utilities.Utils


class NasdaqShareOrdersManager:
    def __init__(self):
        self.session = requests.Session()

    @staticmethod
    def get_nasdaq_trade_order(ticker):

        ticker = 'AAPL'
        print('Getting {} data'.format(ticker))

        res = {'nlsTime': [], 'nlsPrice': [], 'nlsShareVolume': []}

        limit = 999999999
        repeating = False
        curr_data = ''
        curr_time = datetime.datetime.strptime('09:30', '%H:%M')

        # Start session
        session = requests.Session()
        options = Options()
        options.add_argument('--headless')

        while not repeating:

            curr_hour = str(curr_time.hour)
            if curr_time.hour < 10:
                curr_hour = '0' + curr_hour

            curr_min = str(curr_time.minute)
            if curr_time.minute < 10:
                curr_min = '0' + curr_min

            url_extension = '/api/quote/{}/realtime-trades?&limit={}&fromTime={}'.format(ticker, limit, curr_hour + ':'
                                                                                         + curr_min)
            data_cmd = 'https://api.nasdaq.com' + url_extension
            header = {'authority': 'api.nasdaq.com',
                      'path': url_extension}

            data = []
            jdata = {'data': None}

            # Set header attribute based on operating system
            if os.name == 'nt':
                # Windows
                header['user-agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like ' \
                                       'Gecko) Chrome/92.0.4515.107 Safari/537.36'

                while True:
                    try:
                        data = session.get(data_cmd, headers=header).text
                        jdata = json.loads(data)
                        break
                    except Exception as e:
                        print(e)
                        time.sleep(5)
                        continue

            else:
                # Assume linux
                header['user-agent'] = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) ' \
                                       'Chrome/90.0.4430.212 Safari/537.36'

                with webdriver.Chrome(ChromeDriverManager().install()) as driver:

                    while True:
                        try:
                            driver.get(data_cmd)
                            data = driver.find_element_by_tag_name('pre').text
                            jdata = json.loads(data)
                            break
                        except Exception as e:
                            print(e)
                            time.sleep(5)
                            continue

            if data == curr_data:
                repeating = True
                continue
            curr_data = data

            # For invalid data requests
            if not jdata['data'] or not jdata['data']['rows']:
                repeating = True
                continue

            # Iterate through data rows in reverse to keep older times at the end
            for row in reversed(jdata['data']['rows']):

                res['nlsTime'].append(row['nlsTime'])
                res['nlsPrice'].append(row['nlsPrice'])
                res['nlsShareVolume'].append(row['nlsShareVolume'])

            # Increase timeframe
            curr_time += datetime.timedelta(minutes=30)

        return res

    @staticmethod
    def get_nasdaq_trade_orders(tickers, curr_date):

        res = {}
        for ticker in tickers:

            # Check if file already exists
            filename = Utils.get_full_path_from_file_date(curr_date, '{}_share_orders_'.format(ticker),
                                                          '.csv', '../data/Nasdaq Share Order Flow/', True)
            if path.exists(filename):
                res[ticker] = (pd.DataFrame(), filename)
                continue

            res[ticker] = (pd.DataFrame(NasdaqShareOrdersManager.get_nasdaq_trade_order(ticker)), filename)

        return res

    @staticmethod
    def write_nasdaq_trade_orders(curr_date=None, tickers=None):

        if not curr_date:
            curr_date = Utils.datetime_to_time_str(datetime.datetime.today())

        print('Getting Nasdaq Share Orders for {}.'.format(curr_date))

        if not tickers:
            tickers = Utils.get_tickers_from_csv('../doc/sp-500.csv')

        # Chunk the data to save on memory
        tick_limit = 100
        tickers_chunks = [tickers[t:t + tick_limit] for t in range(0, len(tickers), tick_limit)]

        # Iterate through each chunk to write out data
        data = []
        files = []
        for ts in tickers_chunks:

            data = NasdaqShareOrdersManager.get_nasdaq_trade_orders(ts, curr_date)

            # Iterate through tickers and write to csvs
            for ticker in ts:

                filepath = data[ticker][1]
                files.append(filepath)

                if not data[ticker][0].empty:
                    Utils.write_dataframe_to_csv(data[ticker][0], filepath)

        sub_dir = 'Share Order Flow/' + '/'.join(files[0].split('/')[3:-1])
        Utils.upload_files_to_gdrive(files, sub_dir)

        return data


def main():
    NasdaqShareOrdersManager.write_nasdaq_trade_orders()


if __name__ == '__main__':
    main()
