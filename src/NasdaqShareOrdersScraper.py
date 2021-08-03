import requests
import importlib
import json
import time
import datetime
import pandas as pd
from os import path


Utils = importlib.import_module('utilities').Utils


class NasdaqShareOrdersManager:

    @staticmethod
    def get_nasdaq_trade_order(ticker):

        print('Getting {} data'.format(ticker))

        res = {'nlsTime': [], 'nlsPrice': [], 'nlsShareVolume': []}

        limit = 999999999
        repeating = False
        curr_data = ''
        curr_time = datetime.datetime.strptime('09:30', '%H:%M')

        while not repeating:

            curr_hour = str(curr_time.hour)
            if curr_time.hour < 10:
                curr_hour = '0' + curr_hour

            curr_min = str(curr_time.minute)
            if curr_time.minute < 10:
                curr_min = '0' + curr_min

            url_extension = '/api/quote/{}/realtime-trades?&limit={}&fromTime={}'.format(ticker, limit, curr_hour + ':' +
                                                                                         curr_min)
            data_cmd = 'https://api.nasdaq.com' + url_extension
            header = {'authority': 'api.nasdaq.com',
                      'path': url_extension,
                      'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                                    'Chrome/92.0.4515.107 Safari/537.36',
                      'cookie': '_ga=GA1.2.738673769.1627537799; recentlyViewedList=AMC|Stocks; OptanonAlertBoxClosed=2'
                                '021-07-29T05:51:20.068Z; AKA_A2=A; ak_bmsc=DAC8EC27D832FB445F194E89BA57E6F7~0000000000'
                                '00000000000000000000~YAAQhowbuMf/fLF6AQAAPUD/CQwYVQu5vz1MWai/wI7IwAVuDhglzIyp0S2kmXhJF'
                                'TmVcCf7NeVtDaoPSWLuwNqMNQ13ZOI87KMItpGkT3C7K/hI4tqWzVeTN3lMtbdH9VpTTxdtDC7r/Ptazhr/FdX'
                                'FzkYZOA6A+8vlHmldKlCO9es6ciSlXEttT5nMgVIQ6gqBmsD3UtmP2TgHLZ+0KrQ+AxdbOHwFKd5Mh0hU1cioG'
                                '9Ch5BcJtJV2IHB8peXyUnJADYi5pnbWTkNYguChAYTc3TmW0t+o8rRr365XNO4yQm7OT+NjseYcBOrt/W1/P/T'
                                'C99U+FpRs/2AK6l08SW1o3au0ih0GqZtBlRhZmOZmNdJesPbe75tR7sqFfal4uZTBafqPtow1vy8fWCAzeZbSB'
                                'iB0V1/JRgr0JJDdBrFPgFe/; _gid=GA1.2.2008760097.1627960329; _gat=1; entryUrl=https://ww'
                                'w.nasdaq.com/market-activity/stocks/amc/latest-real-time-trades; entryReferringURL=htt'
                                'ps://www.nasdaq.com/market-activity/stocks/amc/real-time; OptanonConsent=isGpcEnabled='
                                '0&datestamp=Mon+Aug+02+2021+20:12:09+GMT-0700+(Pacific+Daylight+Time)&version=6.20.0&i'
                                'sIABGlobal=false&hosts=&landingPath=NotLandingPage&groups=C0004:0,C0001:1,C0003:0,C000'
                                '2:0&AwaitingReconsent=false&geolocation=US;CA; RT="z=1&dm=nasdaq.com&si=0b2a4886-7e75-'
                                '4879-ad91-5598c2e66c17&ss=kru1okq2&sl=1&tt=tm&bcn=//173e255e.akstat.io/&ld=1fv4lr"; bm'
                                '_mi=80018A64474C629F3ADC3444857903DB~ZeO2lbpajvkwN0uTyZqe15nLxsJB/hVTWXeyipf0N3zFj9srW'
                                '8wFIA0pCB3HWi6a2rcG8+zL6bk4mPRF0+Cf/XtcYeuEAwtKYzWOXhsWPY+AGiY9Wd2Q/wcJ8Gcg9hxrWPmMUUh'
                                'ZgEljbtORQ4hrAEL5cDIVriwVV2GUqvG8tlxpT8MSPvqvzzAJAomjCPTxNNllsSNi+aMDLdJRAUIga0uX2X6kZ'
                                'NEZ2ZXv8cTkxbxnyBCrWzLg3scCmgAiisWCSEMiIEuUwF1qBLkhIY+QNFkQhg/Qc/w9XP0wASEwvtU=; NSC_W'
                                '.OEBR.DPN.7070=ffffffffc3a0f70e45525d5f4f58455e445a4a422dae; bm_sv=A12268D33318D7E814F'
                                'BA4032B570BDB~eQdHNzlGQP9bQd8YuRBN/7NNIDBASPKpKiWrKz8tBcgTPeCLhPyi/KiCtRoondrVaCx8wLnS'
                                'iTpTVTX/mMfQu90YB+OnOWBwo5qQtOY09fc9yDfYQDMA7IhFEuHbBsmrCqI59WUnjF07chLFsUjakkqzYsUXLW'
                                'o6pi+gg2yH2pw='}

            data = []
            jdata = {'data': None}
            while True:
                try:
                    data = requests.get(data_cmd, headers=header)
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
