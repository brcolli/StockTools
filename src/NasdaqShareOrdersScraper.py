import requests
import importlib
import json
import time
import datetime
import pandas as pd


Utils = importlib.import_module('utilities').Utils


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
                  'cookie': 'recentlyViewedList=AMD|Stocks,AAPL|Stocks,NDAQ|Stocks,CRM|Stocks,AMC|Stocks; visid_incap_'
                            '2170999=AUmeTSAJR9iHbuH7NDYK2V2BG2AAAAAAQUIPAAAAAAARn/KCFFum59zF2XJESLJt; incap_ses_1163_'
                            '2170999=FDajM6YP1WZckMnOkM4jEF6BG2AAAAAAg286bAKJxp8xhHGwzkcE9g==; bm_mi=8261F0F97854EFC3E'
                            'A27B7BD8952C115~Iocy2Bqx/0lL/0wb215WggylM9SqS5doA5U1arCMEvcQ0nhvlydEIqkH9hzsC26ds4mX5ymSH'
                            '2JibJsv8ghqax3d6r6zXcy5yGZ2XSYtv+BTLAcBOAENKbSXCK6kX7EshI+HDZBiZMCg5nY4x1hWDRBpB3Zp0mNFEm'
                            'V7HFfp30IliewQDjadY6caN96VLH31ntfAdMeTECK8o2wbfUocgMQbvk+Geb8JEPoo6cTUOnQNO3g1FsoM2EPOsPO'
                            '4wOL8; bm_sv=86C8803AA28B555F82C5561180C57C4D~enV9al4MrEyTm6UC5NTcRoZ8WlrETeGJlo4CJ2n1aNv'
                            'S1oEQnKc6xqrgw9mrEi94iQJOKmgjVjo1ztV6P6Hxnb7EKU/zNfYsTirjZ3dMGq/6NQ81whPAqnphteVvDVErFY4O'
                            '4e6lSeTxs9gGRwP8lQFbdIR6/dPgIA7qdyH7WpY=; ak_bmsc=A6FA0CB8CA5A2FF9BB47F7EB78BA661FB81B8CB'
                            '58A350000787C1B6054EADD42~pla//dtKpllOD61TscYyNh2S2Cu92iiEfC86YpNYoAnSOI93/miqXrVUiyHOhEP'
                            'uwy44M4NNaoZBM7NXlwPM49U+7iNP/+1bj1PkrbsdRqBLyqeG10LZjICYIEknCBrDqvC/0sha9Xa3D/B30ZfXai0m'
                            'PsK+ZKTxDhPzZinwIGHNn+bCpQaVo1DbcRNswl/jBQp/GQSFghcfJ0EkIIBgV0IUwj43Zo9MrmgxUgLv5W8SqJItI'
                            'C1aw95bpK37TkyWL1BL3gp8ko4GvnIbxzCvsgPVZVm7LNhWu+cTmXk8yG6uPhX7QUvMmu0hnb6+7UEwJR; RT="z='
                            '1&dm=nasdaq.com&si=0lfb4hze973b&ss=kkqdph10&sl=c&tt=36q&obo=7&rl=1"; NSC_W.OEBR.DPN.7070='
                            'ffffffffc3a0f73345525d5f4f58455e445a4a422dae'}

        while True:
            try:
                data = requests.get(data_cmd, headers=header)
                break
            except:
                time.sleep(5)
                continue

        if data.text == curr_data:
            repeating = True
            continue
        curr_data = data.text

        jdata = json.loads(data.text)

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


def get_nasdaq_trade_orders(tickers):

    res = {}
    for ticker in tickers:
        res[ticker] = pd.DataFrame(get_nasdaq_trade_order(ticker))

    return res


def main():

    tickers = ['TIF']
    #tickers = Utils.get_tickers_from_csv('../doc/sp-500.csv')
    data = get_nasdaq_trade_orders(tickers)

    # Iterate through tickers and write to csvs
    for ticker in tickers:

        curr_date = Utils.datetime_to_time_str(datetime.datetime.today())
        filename = Utils.get_full_path_from_file_date(curr_date, '{}_share_orders_'.format(ticker),
                                                      '.csv', '../data/Share Orders/')

        Utils.write_dataframe_to_csv(data[ticker], filename)


if __name__ == '__main__':
    main()