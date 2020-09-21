import datetime
import pandas as pd
from pandas.tseries.offsets import BDay


class Utils:

    @staticmethod
    def get_last_trading_day():
        return datetime.datetime.today() - BDay(1)

    @staticmethod
    def get_proper_date_format(date):
        return str(date).split(' ')[0]

    @staticmethod
    def get_following_week_range():

        # Current week, starting on Sunday and ending on Saturday
        today = datetime.date.today()

        start = (today + datetime.timedelta(days=7)) - \
            datetime.timedelta(days=(today+datetime.timedelta(days=1)).weekday())
        end = start + datetime.timedelta(days=6)

        return start, end

    @staticmethod
    def time_str_to_datetime(time_str):
        return datetime.datetime.strptime(time_str, '%Y/%m/%d')

    '''
    Writes a pandas DataFrame to a csv
    '''
    @staticmethod
    def write_dataframe_to_csv(df, filename):

        if df.empty:
            print('Data is empty. Please try again.')
        else:
            print('Writing data to ' + filename + '...')

            # Attempt to write to csv
            try:
                df.to_csv(filename)
            except:
                print('Could not open ' + filename + '. Is the file open?')

    @staticmethod
    def load_csv_to_dataframe(filename):
        return pd.read_csv(filename, index_col=[0])
