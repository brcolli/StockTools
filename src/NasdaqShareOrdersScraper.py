import requests
import importlib
import json
import time
import datetime
import pandas as pd
from os import path
import os

Utils = importlib.import_module('utilities').Utils


class NasdaqShareOrdersManager:
    def __init__(self):
        self.session = requests.Session()

    def get_nasdaq_trade_order(self, ticker):

        print('Getting {} data'.format(ticker))

        res = {'nlsTime': [], 'nlsPrice': [], 'nlsShareVolume': []}

        limit = 999999999
        repeating = False
        curr_data = ''
        curr_time = datetime.datetime.strptime('09:30', '%H:%M')

        # Start session
        session = requests.Session()
        cookies = session.cookies

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

            # Set header attribute based on operating system
            if os.name == 'nt':
                # Windows
                header['user-agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like ' \
                                       'Gecko) Chrome/92.0.4515.107 Safari/537.36'
            else:
                # Assume linux
                header['user-agent'] = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) ' \
                                       'Chrome/90.0.4430.212 Safari/537.36'
                
            data = []
            jdata = {'data': None}
            while True:
                try:
                    data = session.get(data_cmd, headers=header, cookies=cookies)
                    jdata = json.loads(data.text)
                    break
                except:
                    time.sleep(5)
                    continue

            if data.text == curr_data:
                repeating = True
                continue
            curr_data = data.text

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

    def get_nasdaq_trade_orders(self, tickers, curr_date):

        res = {}
        for ticker in tickers:

            # Check if file already exists
            filename = Utils.get_full_path_from_file_date(curr_date, '{}_share_orders_'.format(ticker),
                                                          '.csv', '../data/Nasdaq Share Order Flow/', True)
            if path.exists(filename):
                res[ticker] = (pd.DataFrame(), filename)
                continue

            res[ticker] = (pd.DataFrame(self.get_nasdaq_trade_order(ticker)), filename)

        return res

    def write_nasdaq_trade_orders(self, curr_date=None, tickers=None):

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

            data = self.get_nasdaq_trade_orders(ts, curr_date)

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

    nso = NasdaqShareOrdersManager()

    nso.write_nasdaq_trade_orders()


if __name__ == '__main__':
    main()
