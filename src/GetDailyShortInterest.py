import sys
import importlib
import numpy as np
import pandas as pd
from os import path
import datetime

try:
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename
except ImportError:
    pass

Utils = importlib.import_module('utilities').Utils
TCM = importlib.import_module('TdaClientManager').TdaClientManager
Sqm = importlib.import_module('SqliteManager').SqliteManager


class ShortInterestManager:

    # Writes to file, but ignores added newlines
    @staticmethod
    def write_data_to_file_no_newline(filename, data):
        with open(filename, 'a', newline='') as f:
            f.write(data)

    # From a selected file, load and reformat, then save to csv
    def load_short_interest_text_and_write_to_csv(self, filename):

        sel_file = filename.split('/')[-1]
        input_name = sel_file.split('.')[0]
        output = '../data/' + input_name + '.csv'

        f1 = open(filename, 'r')

        data = f1.read()
        data = Utils.replace_line_to_comma(data)

        f1.close()

        self.write_data_to_file_no_newline(output, data)

    # Uses util function to reformat latest trading day
    @staticmethod
    def get_latest_trading_day():
        return Utils.datetime_to_time_str(Utils.get_last_trading_day())

    @staticmethod
    def get_optimal_past_history(tcm, tickers, start_date, end_date=None):

        # Try to pull data from the database first
        ps = tcm.get_past_history(tickers, start_date, end_date)

        return ps

    @staticmethod
    def get_vix_close(tcm):

        vk = '$VIX.X'
        vq = tcm.get_quotes_from_tda([[vk]])

        return vq[vk]['closePrice']

    def get_past_short_vol(self, tickers, tcm, prev_date, prev_data, short_file_prefix):

        # Get data from past if exists
        file_root = '../data/'
        files = Utils.find_file_pattern(short_file_prefix + prev_date + '*', file_root, recurse=True)

        if files:

            latest_data = Utils.get_full_path_from_file_date(prev_date, short_file_prefix, '.csv')
            latest_df = pd.read_csv(latest_data)

            try:
                latest_df = latest_df.set_index('Symbol')
            except:
                latest_df = latest_df

            # Convert to float
            cols = latest_df.columns[latest_df.dtypes.eq('object')]
            latest_df[cols] = latest_df[cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            latest_df = latest_df.replace(np.nan, 0)

            prev_short_perc = latest_df['Shares Traded Short/Float %']
            prev_vol_perc = latest_df['Total Volume']

        else:

            prev_datetime = Utils.time_str_to_datetime(prev_date)
            th = self.get_optimal_past_history(tcm, tickers, prev_datetime)

            prev_data = prev_data.loc[tickers]

            prev_short_perc = prev_data['Shares Traded Short']
            prev_vol_perc = []
            for key in th.keys():
                if len(th[key]) > 0:
                    prev_vol_perc.append(th[key][-1]['volume'])
                else:
                    prev_vol_perc.append(-1)

        return prev_short_perc, prev_vol_perc

    def cleanup_quotes_df(self, tcm, qs_df):

        # Clean up possible 0 values from TDA quotes
        bad_quote_tickers = qs_df[(qs_df['totalVolume'] == 0) | (qs_df['openPrice'] == 0) |
                                  (qs_df['regularMarketLastPrice'] == 0)].index.values

        if len(bad_quote_tickers) > 0:

            tick_limit = 100  # IEX's limit for basic query
            tickers_chunks = [bad_quote_tickers[t:t + tick_limit].tolist()
                              for t in range(0, len(bad_quote_tickers), tick_limit)]
            new_quotes = tcm.get_quotes_from_iex(tickers_chunks)

            # Just parse out possible bad entries
            new_vals = Utils.reduce_double_dict_to_one(new_quotes, ['latestVolume', 'open', 'close'])

            # If still have bad values, replace with -1
            for val in new_vals:
                for nt in val.keys():
                    if val[nt] is None:
                        val[nt] = -1

            qs_df['totalVolume'].update(pd.Series(new_vals[0]))
            qs_df['openPrice'].update(pd.Series(new_vals[1]))
            qs_df['regularMarketLastPrice'].update(pd.Series(new_vals[2]))

        # Add VIX close
        qs_df['VIX Close'] = self.get_vix_close(tcm)

        return qs_df

    def generate_quotes_df(self, tcm, tickers_chunks):

        # Get some TDA data
        qs = tcm.get_quotes_from_tda(tickers_chunks)
        qs_df = pd.DataFrame(qs).transpose()  # Convert to dataframe

        return self.cleanup_quotes_df(tcm, qs_df)

    @staticmethod
    def generate_fundamentals_df(tcm, tickers_chunks):

        fs = tcm.get_fundamentals_from_tda(tickers_chunks)
        fs_df = pd.DataFrame(fs).transpose()

        fs_df.replace(0, np.nan, inplace=True)

        # Populate with stats from saved table
        stats_table = Sqm('../data/Databases/yahoo_stats.sqlite')
        stats = pd.read_sql_query('SELECT * from fundamentals', stats_table.connection)
        stats = stats.set_index('Symbol')

        fs_df['Floats'] = stats['Floats']
        fs_df['sector'] = stats['sector']

        return fs_df

    def generate_past_df(self, short_file_prefix, tcm, tickers, valid_dates, should_filter=False):

        # Add VIX to tickers list
        vk = '$VIX.X'
        tickers = tickers + [vk]

        # Go through each valid date to check if the csv is saved. If all dates are saved, don't call TD API
        not_all_done = False
        ps = {}
        for vd in valid_dates:

            file_root = '../data/'
            files = Utils.find_file_pattern(short_file_prefix + vd + '*', file_root, recurse=True)

            # Either check if csv is found for date or the last day in array matches most recent trading day
            if files:

                vd_df = pd.read_csv(files[0])

                # Rename some columns
                vd_df = vd_df.rename(columns={'Open': 'open', 'Close': 'close', 'Total Volume': 'volume',
                                              'Date': 'datetime'})
                vd_df = vd_df.set_index('Symbol')
                ps[vd] = pd.DataFrame(vd_df)

            elif vd == valid_dates[-1] and vd == Utils.datetime_to_time_str(Utils.get_last_trading_day()):

                # Combine tickers into chunks
                tick_limit = 300  # TDA's limit for basic query
                tickers_chunks = [tickers[t:t + tick_limit] for t in range(0, len(tickers), tick_limit)]

                quotes = tcm.get_quotes_from_tda(tickers_chunks)

                vd_df = pd.DataFrame(quotes).transpose()  # Convert to dataframe
                vd_df = self.cleanup_quotes_df(tcm, vd_df)

                # Reformat to dataframe
                temp = {'open': vd_df['openPrice'], 'close': vd_df['closePrice'],
                        'volume': vd_df['totalVolume'],
                        'datetime': vd,
                        'VIX Close': 'VIX Close', 'Symbol': 'Symbol'}
                ps[vd] = pd.DataFrame(temp)
            else:
                # If even one date fails, call all get_past_history
                not_all_done = True
                break

        if not not_all_done:
            return ps

        ps = self.get_optimal_past_history(tcm, tickers, Utils.time_str_to_datetime(valid_dates[0]),
                                           Utils.time_str_to_datetime(valid_dates[-1]))

        # Sort into a dictionary of historical data dataframes
        ps_dfs = {}

        for i in range(len(valid_dates)):

            temp = {}
            vc = -1
            vd = valid_dates[i]

            # Convert current valid date to epoch milli time
            et = Utils.datetime_to_epoch(Utils.time_str_to_datetime(vd))
            defaults = {'open': 0, 'close': 0, 'volume': 0, 'datetime': et}
            for key, val in ps.items():

                # If didn't have a large enough return, must get remaining missing values
                try:
                    curr_ticker_data = val[i]
                except (KeyError, IndexError):
                    curr_ticker_data = defaults
                    val.insert(i, curr_ticker_data)

                # Check if date matches the order of valid dates
                if Utils.datetime_to_time_str(Utils.epoch_to_datetime(curr_ticker_data['datetime'])) != vd:
                    curr_ticker_data = defaults
                    val.insert(i, curr_ticker_data)

                temp[key] = curr_ticker_data

                # Store VIX close
                if vc == -1 and key == vk:
                    vc = val[i]['close']

            ps_df = pd.DataFrame(temp).transpose()

            # Filter out tickers that don't meet volume and value criteria
            if should_filter:
                ps_df = ps_df[ps_df['volume'] > 5E5]
                ps_df = ps_df[ps_df['close'] > 5]

            ps_df.replace(0, np.nan, inplace=True)

            # Add VIX to dataframe
            ps_df['VIX Close'] = vc

            ps_dfs[valid_dates[i]] = ps_df

        return ps_dfs

    @staticmethod
    def update_short_df_with_data(df, qs, fs, prev_short_perc, prev_vol_perc):

        # Fill in some quote columns
        df['Exchange'] = fs['exchange']
        df['Sector'] = fs['sector']
        df['Total Volume'] = qs['totalVolume']
        df['Open'] = qs['openPrice']
        df['High'] = qs['highPrice']
        df['Low'] = qs['lowPrice']
        df['Close'] = qs['regularMarketLastPrice']
        df['VIX Close'] = qs['VIX Close']
        df['Days to Cover'] = df['ShortVolume'] / fs['vol10DayAvg']

        # Calculate short interest %
        short_int = df['ShortVolume'] / fs['Floats']
        df['Shares Traded Short/Float %'] = short_int

        short_denom = short_int.sub(prev_short_perc).fillna(short_int)
        df['Shares Traded Short/Float % Delta'] = short_denom.div(prev_short_perc)
        df['Total Volume % Delta'] = df['Total Volume'].sub(prev_vol_perc).fillna(df['Total Volume']).div(prev_vol_perc)

        # Calculate % close and open changes
        df['Close % Delta'] = qs['regularMarketPercentChangeInDouble'] / 100
        df['Open-Close % Delta'] = (df['Close'] - df['Open']) / df['Open']

        # Add outstanding shares
        df['Total Volume/Float %'] = df['Total Volume'] / fs['Floats']

        # Only take tickers whose volume delta isn't 0
        df = df.dropna()
        df = df.fillna(0)

        # Rename some columns
        df = df.rename(columns={'ShortVolume': 'Shares Traded Short', 'TotalVolume': 'Finra Volume'})

        return df

    @staticmethod
    def update_short_df_with_past_data(df, ps, fs, date):

        # Sort dates and get old and new data
        dates = list(ps.keys())
        dates.sort()

        old_date = dates[dates.index(date) - 1]
        old_data = ps[old_date]
        curr_data = ps[date]

        # Fill in some data columns
        df['Exchange'] = fs['exchange']
        df['Sector'] = fs['sector']
        df['Total Volume'] = curr_data['volume']
        df['Open'] = curr_data['open']
        df['High'] = curr_data['high']
        df['Low'] = curr_data['low']
        df['Close'] = curr_data['close']
        df['VIX Close'] = curr_data['VIX Close']
        df['Days to Cover'] = df['ShortVolume'] / fs['vol10DayAvg']

        # Calculate short interest %
        short_int = df['ShortVolume'] / fs['Floats']
        df['Shares Traded Short/Float %'] = short_int

        short_denom = short_int.sub(old_data['Shares Traded Short/Float %']).fillna(short_int)
        df['Shares Traded Short/Float % Delta'] = short_denom.div(old_data['Shares Traded Short/Float %'])
        df['Total Volume % Delta'] = df['Total Volume'].sub(old_data['volume']).fillna(
            df['Total Volume']).div(old_data['volume'])

        # Calculate % close and open changes
        df['Close % Delta'] = (curr_data['close'] - old_data['close']) / old_data['close']
        df['Open-Close % Delta'] = (df['Close'] - df['Open']) / df['Open']

        # Add outstanding shares
        df['Total Volume/Float %'] = df['Total Volume'] / fs['Floats']

        # Only take tickers whose volume delta isn't 0
        df = df.dropna()
        df = df.fillna(0)

        # Rename some columns
        df = df.rename(columns={'ShortVolume': 'Shares Traded Short', 'TotalVolume': 'Finra Volume'})

        return df

    def get_past_df(self, short_file_prefix, valid_dates, texts, tickers=None):

        tcm = TCM()
        ps_dfs = None
        fs = None
        dfs = {}

        for i in range(len(valid_dates)):

            text = texts[i]
            df = Utils.regsho_txt_to_df(text)

            # Get list of tickers, separated into even chunks by TDA limiter
            if not tickers:
                tickers = df['Symbol'].tolist()

            tick_limit = 300  # TDA's limit for basic query
            tickers_chunks = [tickers[t:t + tick_limit] for t in range(0, len(tickers), tick_limit)]

            if ps_dfs is None:
                ps_dfs = self.generate_past_df(short_file_prefix, tcm, tickers, valid_dates)

            # Set new index
            df = df.set_index('Symbol')

            # Get fundamental data
            if fs is None:
                fs = self.generate_fundamentals_df(tcm, tickers_chunks)

            # Set short data
            short_int = df['ShortVolume'] / fs['Floats']
            ps_dfs[valid_dates[i]]['Shares Traded Short/Float %'] = short_int

            # Skip added date
            if i >= 1:
                df = self.update_short_df_with_past_data(df, ps_dfs, fs, valid_dates[i])

            dfs[valid_dates[i]] = df

        return dfs

    def get_today_df(self, prev_day, texts, short_file_prefix, tickers=None, should_filter=False):

        df = Utils.regsho_txt_to_df(texts[1])
        past_df = Utils.regsho_txt_to_df(texts[0])

        # Get list of tickers, separated into even chunks by TDA limiter
        if not tickers:
            tickers = df['Symbol'].tolist()

        tick_limit = 300  # TDA's limit for basic query
        tickers_chunks = [tickers[t:t + tick_limit] for t in range(0, len(tickers), tick_limit)]

        # Set new index
        df = df.set_index('Symbol')
        past_df = past_df.set_index('Symbol')

        tcm = TCM()
        qs_df = self.generate_quotes_df(tcm, tickers_chunks)
        fs_df = self.generate_fundamentals_df(tcm, tickers_chunks)

        # Drop symbols with missing data by joining on matching symbols
        qs_syms = qs_df.index.tolist()
        fs_syms = fs_df.index.tolist()

        # Because getting quotes is inconsistent, find missing tickers between quotes and funds and re-get quotes
        excess_tickers = list(np.setdiff1d(fs_syms, qs_syms))

        excess_quotes = tcm.get_quotes_from_tda([excess_tickers])
        eq_df = pd.DataFrame(excess_quotes).transpose()  # Convert to dataframe
        eq_df = self.cleanup_quotes_df(tcm, eq_df)

        # Append cleaned new quotes
        qs_df.append(eq_df)
        qs_df = qs_df.sort_index()

        # Filter on volume minimum and open price minimum
        if should_filter:
            qs_df = qs_df[qs_df['totalVolume'] > 5E5]
            qs_df = qs_df[qs_df['regularMarketLastPrice'] > 5]

        # Remove any remaining excess tickers
        qs_syms = qs_df.index.tolist()

        excess_tickers = list(np.setdiff1d(qs_syms, fs_syms))
        qs_df = qs_df.drop(excess_tickers)
        qs_syms = qs_df.index.tolist()

        fs_df = fs_df.loc[qs_syms]
        df = df.reindex(index=qs_syms)

        (prev_short_perc, prev_vol_perc) = self.get_past_short_vol(qs_syms, tcm, prev_day,
                                                                   past_df, short_file_prefix)

        df = self.update_short_df_with_data(df, qs_df, fs_df, prev_short_perc, prev_vol_perc)

        return df

    @staticmethod
    def get_regsho_csvs_from_dates(short_file_prefix, url, ymd, ymd2=''):

        outputs = []
        valid_dates = []
        texts = []
        date_range = Utils.get_bd_range(ymd, ymd2)

        # Get one extra day to the start, for historical comparison
        date_range = [Utils.datetime_to_time_str(
            Utils.get_previous_trading_day_from_date(Utils.time_str_to_datetime(date_range[0])))] + date_range

        for date in date_range:

            filename = short_file_prefix + date
            out = Utils.get_full_path_from_file_date(date, short_file_prefix, '.csv')

            data = Utils.get_file_from_url(url, filename + '.txt')
            text = Utils.replace_line_to_comma(data)

            # If date not found, find next most recent date
            if '404 Not Found' in text:

                # Get most recent past trading day
                past_td = Utils.datetime_to_time_str(
                    Utils.get_previous_trading_day_from_date(Utils.time_str_to_datetime(date)))
                ymd = past_td

                filename = short_file_prefix + past_td
                out = Utils.get_full_path_from_file_date(past_td, short_file_prefix, '.csv')

                data = Utils.get_file_from_url(url, filename + '.txt')
                text = Utils.replace_line_to_comma(data)

            valid_dates.append(date)
            texts.append(text)
            outputs.append(out)

        return ymd, valid_dates, texts, outputs

    # Gets file from regsho consolidated short interest using a YYYYMMDD format and write to csv
    def get_regsho_daily_short_to_csv(self, ymd, ymd2=''):

        url = 'http://regsho.finra.org'
        short_file_prefix = 'CNMSshvol'

        # Place files to correct subdirectories
        out = Utils.get_full_path_from_file_date(ymd, short_file_prefix, '.csv')

        # Check if date already saved
        if path.exists(out):
            return [out]

        (ymd, valid_dates, texts, outputs) = self.get_regsho_csvs_from_dates(short_file_prefix, url, ymd, ymd2)

        if not outputs:
            return ['']

        # Check if date passed is current day. If not, cannot use quotes
        if Utils.is_it_today(ymd):

            print('Getting daily short interest for {}.'.format(datetime.datetime.today().date()))

            df = self.get_today_df(valid_dates[0], texts, short_file_prefix)
            Utils.write_stock_data_dataframe_to_sqlite(df)
            Utils.write_dataframe_to_csv(df, outputs[1])
        else:

            dfs = self.get_past_df(short_file_prefix, valid_dates, texts)
            for i in range(1, len(outputs)):
                Utils.write_dataframe_to_csv(dfs[valid_dates[i]], outputs[i])

            return outputs[1:]

        return [outputs[1]]

    # Call function to write latest trading day's short interest to a csv
    def get_latest_short_interest_data(self):

        ltd = self.get_latest_trading_day()

        return self.get_regsho_daily_short_to_csv(ltd)

    def import_short_interest_text_from_selection(self):

        if 'tkinter' in sys.modules:
            Tk().withdraw()
            f_name = askopenfilename()
        else:
            f_name = ''

        self.load_short_interest_text_and_write_to_csv(f_name)

    def get_largest_gainers_from_range(self, ymd1, ymd2, tickers=None, threshold=10):

        url = 'http://regsho.finra.org'
        short_file_prefix = 'CNMSshvol'

        (_, valid_dates, texts, _) = self.get_regsho_csvs_from_dates(short_file_prefix, url, ymd1, ymd2)

        if not tickers:
            dfs = self.get_past_df(short_file_prefix, valid_dates, texts)
        else:
            dfs = self.get_past_df(short_file_prefix, valid_dates, texts, tickers)

        first_pass = True
        prev_date = -1
        gainers = []
        for key, val in dfs.items():

            # Skip first entry
            if first_pass:
                first_pass = False
                continue

            if prev_date == -1:
                prev_date = key
                continue

            for row, cols in val.iterrows():

                prev_data = dfs[prev_date]

                # Check if change is greater than the threshold
                if cols['Close % Delta'] * 100 >= threshold and row in prev_data.index:

                    # Only add if the previous data had a change of less than 3%
                    if abs(prev_data.loc[row]['Close % Delta'] * 100) < 3:
                        gainer = {'Symbol': row}  # Add symbol to data
                        gainer.update(prev_data.loc[row].to_dict())
                        gainers.append(gainer)

            prev_date = key

        # Write results to csv output
        Utils.write_list_of_dicts_to_csv(gainers, '../data/Next Day Pop.csv')

        return gainers


def main(ymd1='', ymd2='', should_upload=True):

    sim = ShortInterestManager()

    if ymd1 == '':
        res = sim.get_latest_short_interest_data()
    else:
        res = sim.get_regsho_daily_short_to_csv(ymd1, ymd2)

    if should_upload:
        sub_dir = 'Daily Short Data/' + '/'.join(res[0].split('/')[2:-1])  # Just get subdirectory path
        Utils.upload_files_to_gdrive(res, sub_dir)


if __name__ == '__main__':
    main()
    #df = Utils.load_csv_to_dataframe('../data/2021/0328-0403/CNMSshvol20210330.csv')
    #Utils.write_stock_data_dataframe_to_sqlite(df)
