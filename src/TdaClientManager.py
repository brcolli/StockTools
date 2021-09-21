from tda.auth import easy_client
from tda.streaming import StreamClient
from iexfinance.stocks import Stock
import robin_stocks as rs
import json
import time
import re
import os
import pandas as pd


"""TdaClientManager

Description:
Used for all API function calls regarding stock data gathering. Primary focus is using the TD Ameritrade API.

Authors: Benjamin Collins
Date: April 22, 2021 
"""


class TdaClientManager:

    """Used for all API function calls regarding stock data gathering. Primary focus is using the TD Ameritrade API.
    """

    def __init__(self):

        """Constructor method, creates a TD API client and stream object.
        """

        #  TODO Remove these hardcoded inputs for an encrypted cloud file
        self.account_id = '275356186'
        self.key = 'FA3ISKLEGYIFXQRSUJQCB93AKXFRGZUK'
        self.callback_url = 'https://localhost:8080'
        self.token_path = '../doc/token'

        if not os.path.exists(self.token_path):
            self.authenticate()

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

        """Authenticates a TD client that has expired. Typical expiration time is 30 days.
        """

        from selenium import webdriver
        from webdriver_manager.chrome import ChromeDriverManager

        with webdriver.Chrome(ChromeDriverManager().install()) as driver:
            self.client = easy_client(api_key=self.key,
                                      redirect_uri=self.callback_url,
                                      token_path=self.token_path,
                                      webdriver_func=lambda: driver)

    @staticmethod
    def get_option_historical_data():

        data = rs.robinhood.get_option_historicals(symbol='AAPL', expirationDate='2021-01-08', strikePrice='130',
                                                   optionType='call')
        print(data)

    def find_options(self, tickers, to_date=None, from_date=None, max_mark=2, max_spread=0.5, min_delta=0.3,
                     max_theta=0.02, max_iv=50, min_oi=100):

        """Finds a list of options for tickers given criteria. With a list of tickers, this will get all current options
        within a date range that meet a criteria.

        :param tickers: A list of tickers to search for options
        :type tickers: list(str)
        :param to_date: End date for range to search for options; defaults to None
        :type to_date: datetime
        :param from_date: Start date for range to search for options; defaults to None
        :type from_date: datetime
        :param max_mark: The maximum value that an options contract was last traded at; defaults to 2
        :type max_mark: float
        :param max_spread: The maximum difference between the ask and bid, will be taken as an absolute value;
                           defaults to 0.5
        :type max_spread: float
        :param min_delta: The minimum value of delta for a contract, will be taken as an absolute value; defaults to 0.3
        :type min_delta: float
        :param max_theta: The maximum value of theta for a contract, will be taken as an absolute value;
                          defaults to 0.02
        :type max_theta: float
        :param max_iv: The maximum value of implied volatility for a contract; defaults to 50
        :type max_iv: float
        :param min_oi: The minimum value of open interest for a contract; defaults to 100
        :type min_oi: int

        :return: A dictionary of :class:`pandas.core.frame.DataFrame` for each ticker,
                 containing possible options contracts of interest; defaults to empty dictionary
        :rtype: dict(str-> :class:`pandas.core.frame.DataFrame`)
        """

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
            opts[ticker] = pd.DataFrame()

            # Iterate through all calls and puts and filter on criteria
            try:
                call_dates = data['callExpDateMap']
                put_dates = data['putExpDateMap']
            except Exception as e:
                print(f'Bad options data for {ticker} due to {e}. Skipping.')
                continue

            #  Collect call and put options that pass the criteria in a dictionary
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

        """Gets daily quotes from IEX Finance given a list of lists of tickers. Requires nested lists as IEX Finance
        has a maximum amount of tickers that can be requested at a given time.

        :param tickers_chunks: A list of lists of tickers, where max chunk size is 100 tickers
        :type tickers_chunks: list(list(str))

        :return: A dictionary of :class:`pandas.core.frame.DataFrame` for each ticker,
                 containing quote data; defaults to empty dictionary
        :rtype: dict(str-> :class:`pandas.core.frame.DataFrame`)
        """

        ret = {}
        for tickers in tickers_chunks:

            #  Continue to iterate on the ticker in each chunk to deal with rate limiting
            while True:

                #  TODO replace token with encrypted cloud file
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

        """Gets daily quotes from TD Ameritrade given a list of lists of tickers. Requires nested lists as TD Ameritrade
        has a maximum amount of tickers that can be requested at a given time.

        :param tickers_list: A list of lists of tickers, where max chunk size is 300 tickers
        :type tickers_list: list(list(str))

        :return: A dictionary of :class:`pandas.core.frame.DataFrame` for each ticker,
                 containing quote data; defaults to empty dictionary
        :rtype: dict(str-> :class:`pandas.core.frame.DataFrame`)
        """

        qs = {}
        for tickers in tickers_list:

            #  Continue to iterate on the ticker in each chunk to deal with rate limiting
            while True:

                r = None
                while True:
                    try:
                        r = self.client.get_quotes(tickers)
                        break
                    except Exception as e:

                        if isinstance(e, InvalidGrantError) and e.args == InvalidGrantError().args:
                            if os.path.exists('../doc/token'):
                                os.remove('../doc/token')
                            self.authenticate()

                        time.sleep(1)
                        continue

                #  Attempt to parse the data
                data = r.json()

                if not data:
                    print('{} quote data not found.'.format(tickers))
                    break

                first_key = list(data.keys())
                first_key = first_key[0]

                if first_key != 'error':
                    break
                else:
                    time.sleep(5)  # Most likely rate limited, wait 5 seconds and try again

            qs.update(data)

        return qs

    def get_fundamentals_from_tda(self, tickers_list):

        """Gets fundamentals from TD Ameritrade given a list of lists of tickers. Requires nested lists as TD Ameritrade
        has a maximum amount of tickers that can be requested at a given time.

        :param tickers_list: A list of lists of tickers, where max chunk size is 300 tickers
        :type tickers_list: list(list(str))

        :return: A dictionary of :class:`pandas.core.frame.DataFrame` for each ticker,
                 containing fundamental data; defaults to empty dictionary
        :rtype: dict(str-> :class:`pandas.core.frame.DataFrame`)
        """

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

        """Gets historical stock data for a ticker given a range of dates. Data includes open, high, low, close, and
        volume for a ticker on a specific day. Calls one ticker at a time as TD API does not support multiple tickers
        for a single call.

        :param tickers: A list of tickers
        :type tickers: list(str)

        :return: A dictionary of :class:`pandas.core.frame.DataFrame` for each ticker,
                 containing fundamental data; defaults to empty dictionary
        :rtype: dict(str-> :class:`pandas.core.frame.DataFrame`)
        """

        tickers_history = {}

        if end_day is None:
            end_day = start_day

        for ticker in tickers:

            #  Loop on a single ticker to ensure data is retrieved and we don't get rate limited out
            while True:

                # Second internal loop to handle possible authentication timeouts
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

                        if e.error == 'invalid_grant':
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
