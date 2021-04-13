from tda.auth import easy_client
from tda.streaming import StreamClient
from iexfinance.stocks import Stock
import json
import time
import re
from oauthlib.oauth2.rfc6749.errors import InvalidGrantError
import os
import pandas as pd


class TdaClientManager:

    def __init__(self):

        self.account_id = '275356186'
        self.key = 'FA3ISKLEGYIFXQRSUJQCB93AKXFRGZUK'
        self.callback_url = 'https://localhost:8080'
        self.token_path = '../doc/token'

        # Setup client
        try:
            self.client = easy_client(api_key=self.key,
                                      redirect_uri=self.callback_url,
                                      token_path=self.token_path)
        except FileNotFoundError:
            self.authenticate()

        # Setup stream
        self.stream = StreamClient(self.client, account_id=self.account_id)

    def authenticate(self):

        from selenium import webdriver
        from webdriver_manager.chrome import ChromeDriverManager

        with webdriver.Chrome(ChromeDriverManager().install()) as driver:
            self.client = easy_client(api_key=self.key,
                                      redirect_uri=self.callback_url,
                                      token_path=self.token_path,
                                      webdriver_func=driver)

    def find_options(self, tickers, to_date=None, from_date=None, max_mark=2, max_spread=0.5, min_delta=0.3,
                     max_theta=0.02, max_iv=50, min_oi=100):

        opts = {}

        # Function defining criteria
        def criteria(mark, spread, delta, theta, iv, oi):

            # Handle NaN strings
            if delta == 'NaN':
                delta = -1
            if theta == 'NaN':
                theta = float('inf')
            if iv == 'NaN':
                iv = float('inf')
            if oi == 'NaN':
                oi = -1
            return (mark <= max_mark and abs(spread) <= max_spread
                    and abs(delta) >= min_delta and abs(theta) <= max_theta
                    and iv <= max_iv and oi >= min_oi)

        for ticker in tickers:

            try:
                if to_date:
                    r = self.client.get_option_chain(ticker, strike_count=100,
                                                     strategy=self.client.Options.Strategy.ANALYTICAL,
                                                     from_date=from_date, to_date=to_date)
                else:
                    r = self.client.get_option_chain(ticker, strike_count=100,
                                                     strategy=self.client.Options.Strategy.ANALYTICAL)

            except Exception as e:
                print('Caught {} for ticker {}.'.format(e, ticker))
                break

            data = r.json()
            opts[ticker] = []

            # Iterate through all calls and puts and filter on criteria
            call_dates = data['callExpDateMap']
            put_dates = data['putExpDateMap']
            opt_list = []

            for call_date, calls in call_dates.items():
                for strike, call_ in calls.items():
                    call = call_[0]
                    if criteria(call['mark'], call['ask']-call['bid'], call['delta'], call['theta'],
                                call['volatility'], call['openInterest']):
                        opt_list.append(call)

            for put_date, puts in put_dates.items():
                for strike, put_ in puts.items():
                    put = put_[0]
                    if criteria(put['mark'], put['ask']-put['bid'], put['delta'], put['theta'],
                                put['volatility'], put['openInterest']):
                        opt_list.append(put)

            opts[ticker] = pd.DataFrame(opt_list)

        return opts

    @staticmethod
    def get_quotes_from_iex(tickers_chunks):

        ret = {}
        for tickers in tickers_chunks:

            while True:
                s = Stock(tickers, token='pk_78c8ddd19775400684a6a51744aaacd6')
                try:
                    iexq = s.get_quote()
                    break
                except Exception as e:

                    r = re.match(r'Symbol (.*) not found\.', e.__str__())

                    # Remove the bad symbol completely and try again
                    if r:
                        bad_sym = r.group(1)
                        tickers.remove(bad_sym)

            for index, row in iexq.iterrows():
                ret[index] = row

        return ret

    def get_quotes_from_tda(self, tickers_list):

        qs = {}
        for tickers in tickers_list:

            while True:

                while True:
                    try:
                        r = self.client.get_quotes(tickers)
                        break
                    except Exception as e:

                        if type(e) is type(InvalidGrantError()) and e.args == InvalidGrantError().args:
                            if os.path.exists('../doc/token'):
                                os.remove('../doc/token')
                            self.authenticate()

                        time.sleep(1)
                        continue

                data = r.json()

                if not data:
                    print('{} quote data not found.'.format(tickers))
                    break

                first_key = list(data.keys())
                first_key = first_key[0]

                if first_key != 'error':
                    break
                else:
                    time.sleep(5)

            qs.update(data)

        return qs

    def get_fundamentals_from_tda(self, tickers_list):

        fs = {}
        for tickers in tickers_list:
            while True:

                r = self.client.search_instruments(tickers, self.client.Instrument.Projection.FUNDAMENTAL)
                data = r.json()
                dkeys = list(data.keys())

                if dkeys[0] != 'error' and len(dkeys) > 0:
                    break
                else:
                    time.sleep(1)

            fs.update(data)

        # Just take fundamental data
        ret = {}
        for key in fs.keys():

            # Add exchange to fundamentals
            temp = fs[key]['fundamental']
            temp['exchange'] = fs[key]['exchange']

            ret[key] = temp

        return ret

    def get_past_history(self, tickers, start_day, end_day=None):

        tickers_history = {}

        if end_day is None:
            end_day = start_day

        for ticker in tickers:

            while True:

                while True:
                    try:
                        r = self.client.get_price_history(ticker,
                                                          start_datetime=start_day,
                                                          end_datetime=end_day,
                                                          period_type=self.client.PriceHistory.PeriodType.MONTH,
                                                          frequency=self.client.PriceHistory.Frequency.DAILY,
                                                          frequency_type=self.client.PriceHistory.FrequencyType.DAILY)
                        break
                    except Exception as e:

                        if e == InvalidGrantError():
                            if os.path.exists('../doc/token'):
                                os.remove('../doc/token')
                            self.authenticate()
                        time.sleep(1)
                try:
                    data = r.json()
                except json.decoder.JSONDecodeError:
                    data = {'candles': [], 'symbol': ticker, 'empty': False}

                try:
                    if data['candles']:
                        tickers_history[ticker] = data['candles']
                    break
                except KeyError:
                    if data['error'] == 'Not found':
                        continue
                    else:
                        time.sleep(5)

        return tickers_history
