from tkinter import Tk
from tkinter.filedialog import askopenfilename
import importlib
import numpy as np
import pandas as pd
from io import StringIO
from os import path


Utils = importlib.import_module('utilities').Utils
TCM = importlib.import_module('TdaClientManager').TdaClientManager


class ShortInterestManager:

    # Reformatting short interest file to a proper csv
    @staticmethod
    def replace_line_to_comma(text):
        text = text.replace(',', '/')
        return text.replace('|', ',')

    # Writes to file, but ignores added newlines
    @staticmethod
    def write_data_to_file_no_newline(filename, data):
        with open(filename, 'a', newline='') as f:
            f.write(data)

    # From a selected file, load and reformat, then save to csv
    @staticmethod
    def load_short_interest_text_and_write_to_csv(filename):

        sel_file = filename.split('/')[-1]
        input_name = sel_file.split('.')[0]
        output = '../data/' + input_name + '.csv'

        f1 = open(filename, 'r')

        data = f1.read()
        data = ShortInterestManager.replace_line_to_comma(data)

        f1.close()

        ShortInterestManager.write_data_to_file_no_newline(output, data)

    # Uses util function to reformat latest trading day
    @staticmethod
    def get_latest_trading_day():
        return Utils.datetime_to_time_str(Utils.get_last_trading_day())

    @staticmethod
    def get_past_short_vol(df, tickers, tcm, ymd, short_file_prefix):

        # Get data from past if exists
        files = Utils.find_file_pattern(short_file_prefix + '*', '../data/')
        prev_data_date = ymd
        if len(files) > 0:

            prev_data_file = files[-1]
            prev_data_file = prev_data_file.split('/')[-1]
            prev_data_date = prev_data_file[len(short_file_prefix):-4]

            if prev_data_date == ymd and len(files) > 1:
                prev_data_file = files[-2]
                prev_data_file = prev_data_file.split('/')[-1]
                prev_data_date = prev_data_file[len(short_file_prefix):-4]

        if prev_data_date != ymd:

            latest_data = '../data/' + short_file_prefix + prev_data_date + '.csv'
            latest_df = pd.read_csv(latest_data)

            try:
                latest_df = latest_df.set_index('Symbol')
            except:
                latest_df = latest_df

            # Convert to float
            cols = latest_df.columns[latest_df.dtypes.eq('object')]
            latest_df[cols] = latest_df[cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            latest_df = latest_df.replace(np.nan, 0)

            prev_short_perc = latest_df['Short Interest Ratio']
            prev_vol_perc = latest_df['TotalVolume']
        else:

            th = tcm.get_past_history(tickers, ymd)
            prev_short_perc = df['TotalVolume']
            prev_vol_perc = []
            for key in th.keys():
                if len(th[key]) > 0:
                    prev_vol_perc.append(th[key][-1]['volume'])
                else:
                    prev_vol_perc.append(-1)

        return prev_short_perc, prev_vol_perc

    @staticmethod
    def generate_quotes_df(tcm, tickers_chunks):

        # Get some TDA data
        qs = tcm.get_quotes_from_tda(tickers_chunks)
        qs_df = pd.DataFrame(qs).transpose()  # Convert to dataframe

        # Clean up possible 0 values from TDA quotes
        bad_quote_tickers = qs_df[(qs_df['totalVolume'] == 0) | (qs_df['openPrice'] == 0) |
                                  (qs_df['regularMarketLastPrice'] == 0)].index.values
        if len(bad_quote_tickers) > 0:

            new_quotes = tcm.get_quotes_from_iex(bad_quote_tickers.tolist())

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

        return qs_df

    @staticmethod
    def generate_fundamentals_df(tcm, tickers_chunks):

        fs = tcm.get_fundamentals_from_tda(tickers_chunks)
        fs_df = pd.DataFrame(fs).transpose()

        fs_df.replace(0, np.nan, inplace=True)

        return fs_df

    @staticmethod
    def regsho_txt_to_df(text):

        # Convert text into a dataframe
        sio = StringIO(text)
        df = pd.read_csv(sio, sep=',')[:-1]
        df = df[df['TotalVolume'] >= 1E6]  # Only take rows with volumes greater than 1 million

        return df

    @staticmethod
    def update_short_df_with_data(df, qs, fs, prev_short_perc, prev_vol_perc):

        # Fill in some quote columns
        df['TotalVolume'] = qs['totalVolume']
        df['Open'] = qs['openPrice']
        df['Close'] = qs['regularMarketLastPrice']
        df['Previous day\'s close change'] = qs['regularMarketPercentChangeInDouble'] / 100

        # Calculate short interest %
        short_int = df['ShortVolume'] / df['TotalVolume']
        df['Short Interest Ratio'] = short_int
        df['Previous short delta'] = short_int.sub(prev_short_perc).fillna(short_int)

        df['Previous volume delta'] = df['TotalVolume'].sub(prev_vol_perc).fillna(df['TotalVolume']).div(prev_vol_perc)

        # Calculate % close
        df['Open/Close change'] = (df['Close'] - df['Open']) / df['Open']

        # Add outstanding shares
        df['Total Volume/Shares Outstanding'] = df['TotalVolume'] / fs['sharesOutstanding']

        df = df.fillna(0)

        return df


    # Gets file from regsho consolidated short interest using a YYYYMMDD format and write to csv
    @staticmethod
    def get_regsho_daily_short_to_csv(ymd):

        url = 'http://regsho.finra.org'
        short_file_prefix = 'CNMSshvol'

        filename = short_file_prefix + ymd
        output = '../data/' + filename + '.csv'

        # Check if date already saved
        if path.exists(output):
            return output

        data = Utils.get_file_from_url(url, filename + '.txt')
        text = ShortInterestManager.replace_line_to_comma(data)

        # If date not found, find next most recent date
        if '404 Not Found' in text:

            # Get most recent past trading day
            past_td = Utils.datetime_to_time_str(Utils.get_recent_trading_day_from_date(Utils.get_yesterday()))

            filename = short_file_prefix + past_td
            output = '../data/' + filename + '.csv'

            data = Utils.get_file_from_url(url, filename + '.txt')
            text = ShortInterestManager.replace_line_to_comma(data)

        df = ShortInterestManager.regsho_txt_to_df(text)

        # Get list of tickers, separated into even chunks by TDA limiter
        tickers = df['Symbol'].tolist()
        tick_limit = 100  # TDA's limit for basic query
        tickers_chunks = [tickers[t:t + tick_limit] for t in range(0, len(tickers), tick_limit)]

        tcm = TCM()
        qs_df = ShortInterestManager.generate_quotes_df(tcm, tickers_chunks)
        fs_df = ShortInterestManager.generate_fundamentals_df(tcm, tickers_chunks)

        # Set new index
        df = df.set_index('Symbol')

        (prev_short_perc, prev_vol_perc) = ShortInterestManager.get_past_short_vol(df, tickers, tcm, ymd,
                                                                                   short_file_prefix)

        df = ShortInterestManager.update_short_df_with_data(df, qs_df, fs_df, prev_short_perc, prev_vol_perc)

        Utils.write_dataframe_to_csv(df, output)
        return output

    # Call function to write latest trading day's short interest to a csv
    @staticmethod
    def get_latest_short_interest_data():
        ltd = ShortInterestManager.get_latest_trading_day()
        return ShortInterestManager.get_regsho_daily_short_to_csv(ltd)

    @staticmethod
    def import_short_interest_text_from_selection():

        Tk().withdraw()
        f_name = askopenfilename()

        ShortInterestManager.load_short_interest_text_and_write_to_csv(f_name)


def main():

    sim = ShortInterestManager
    res = sim.get_latest_short_interest_data()
    Utils.upload_file_to_gdrive(res, 'Daily Short Data')


if __name__ == '__main__':
    main()
