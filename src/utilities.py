import datetime
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.holiday import GoodFriday
from pandas.tseries.offsets import CustomBusinessDay
import random
import requests
from os import path, walk
import fnmatch
from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth
from io import StringIO


class Utils:

    # Define business day
    USFederalHolidayCalendar.rules.pop(7)  # Allow trading on Veterans Day
    USFederalHolidayCalendar.rules.pop(6)  # Allow trading on Columbus Day
    USFederalHolidayCalendar.rules.append(GoodFriday) # Do not allow trading on Good Friday
    BDay = CustomBusinessDay(calendar=USFederalHolidayCalendar())

    @staticmethod
    def upload_file_to_gdrive(filepath, gdrive_dir=''):

        if filepath == '':
            return

        filename = filepath.split('/')[-1]

        GoogleAuth.DEFAULT_SETTINGS['client_config_file'] = '../doc/client_secrets.json'
        gauth = GoogleAuth()

        # Try to load saved client credentials
        gauth.LoadCredentialsFile('../doc/gdrive_creds.txt')
        if gauth.credentials is None:
            # Authenticate if they're not there
            gauth.LocalWebserverAuth()
        elif gauth.access_token_expired:
            # Refresh them if expired
            gauth.Refresh()
        else:
            # Initialize the saved creds
            gauth.Authorize()

        # Save the current credentials to a file
        gauth.SaveCredentialsFile('../doc/gdrive_creds.txt')

        drive = GoogleDrive(gauth)

        file_list = drive.ListFile(
            {'q': "trashed=false", 'includeItemsFromAllDrives': True,
             'supportsAllDrives': True, 'corpora': 'allDrives'}).GetList()

        # folder_id = '12Tcsl57iFVvQyCPr49-VzUfcbHcrXU34'
        # file_list = drive.ListFile({'q': f"parents in '{folder_id}' and trashed=false", 'includeItemsFromAllDrives':
        # True,
        # 'supportsAllDrives': True, 'corpora': 'allDrives'}).GetList()
        folder_id = ''
        for gf in file_list:
            if gf['title'] == gdrive_dir and gf['shared'] is True:
                folder_id = gf['id']
                break

        if folder_id != '':
            f = drive.CreateFile({'title': filename, 'corpora': 'allDrives', 'includeItemsFromAllDrives': True,
                                  'supportsAllDrives': True,
                                  'parents': [{'kind': 'drive#fileLink',
                                               'id': folder_id}]})
        else:
            f = drive.CreateFile({'title': filename})

        f.SetContentFile(filepath)
        f.Upload(param={'supportsAllDrives': True})

        f = None

    # Reformatting short interest file to a proper csv
    @staticmethod
    def replace_line_to_comma(text):
        text = text.replace(',', '/')
        return text.replace('|', ',')

    @staticmethod
    def regsho_txt_to_df(text, vol_lim=-1):

        # Convert text into a dataframe
        sio = StringIO(text)
        df = pd.read_csv(sio, sep=',')[:-1]
        df = df[df['TotalVolume'] >= vol_lim]  # Only take rows with volumes greater than filter

        return df

    # Use an arbitrary day to get finra's ticker list for that day
    @staticmethod
    def get_all_tickers(day):

        url = 'http://regsho.finra.org'
        short_file_prefix = 'CNMSshvol'

        # Import finra csv
        filename = short_file_prefix + day
        data = Utils.get_file_from_url(url, filename + '.txt')
        text = Utils.replace_line_to_comma(data)
        df = Utils.regsho_txt_to_df(text)

        return df['Symbol'].tolist()

    @staticmethod
    def shuffle_list(data):
        random.shuffle(data)
        return data

    @staticmethod
    def get_previous_trading_day_from_date(sel_date):
        return sel_date - Utils.BDay

    @staticmethod
    def get_yesterday():
        return datetime.datetime.today() - datetime.timedelta(days=1)

    @staticmethod
    def get_last_trading_day():
        return datetime.datetime.today() - datetime.timedelta(days=1) + Utils.BDay

    @staticmethod
    def get_proper_date_format(date):
        return str(date).split(' ')[0]

    @staticmethod
    def is_it_today(date):
        return Utils.datetime_to_time_str(datetime.datetime.today()) == date

    @staticmethod
    def get_bd_range(d1, d2):

        if d2 == '':
            return [d1]

        start = Utils.get_previous_trading_day_from_date(Utils.time_str_to_datetime(d1)) + Utils.BDay
        end = Utils.get_previous_trading_day_from_date(Utils.time_str_to_datetime(d2)) + Utils.BDay

        if start > end:
            return [d1]

        res = []

        while start <= end:
            res.append(Utils.datetime_to_time_str(start))
            start += Utils.BDay

        return res

    @staticmethod
    def get_bd_range_from_count(range_len):

        latest = Utils.get_last_trading_day()

        if range_len <= 1:
            return [Utils.datetime_to_time_str(latest)]

        curr_count = 0
        curr_date = latest
        res = []
        while curr_count < range_len:

            prev_date = Utils.get_previous_trading_day_from_date(curr_date)
            res.append(Utils.datetime_to_time_str(prev_date))

            curr_date = prev_date
            curr_count += 1

        return res

    @staticmethod
    def get_following_week_range():

        # Current week, starting on Sunday and ending on Saturday
        today = datetime.date.today()

        start = (today + datetime.timedelta(days=7)) - \
                datetime.timedelta(days=(today + datetime.timedelta(days=1)).weekday())
        end = start + datetime.timedelta(days=6)

        return start, end

    @staticmethod
    def reformat_time_str(time_str):
        return datetime.datetime.strptime(time_str, "%m/%d/%Y").strftime("%Y%m%d")

    @staticmethod
    def time_str_to_datetime(time_str):

        if '/' in time_str:
            return datetime.datetime.strptime(time_str, '%Y/%m/%d')
        else:
            return datetime.datetime.strptime(time_str, '%Y%m%d')

    @staticmethod
    def datetime_to_time_str(ddate):
        ret = Utils.get_proper_date_format(ddate)
        return ret.replace('-', '')

    @staticmethod
    def datetime_to_epoch(ddate):
        return (ddate - datetime.datetime.utcfromtimestamp(0)).total_seconds() * 1000

    @staticmethod
    def epoch_to_datetime(epoch):
        return datetime.datetime.utcfromtimestamp(epoch / 1000)

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

    # Gets a file from a simple URL/FILE format
    @staticmethod
    def get_file_from_url(url, file):

        link = url + '/' + file
        r = requests.get(link, allow_redirects=True)

        return r.text

    @staticmethod
    def find_file_pattern(pattern, root_path, recurse=False):

        result = []
        for root, _, files in walk(root_path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    result.append(path.join(root, name))
            if not recurse:
                break

        return result

    @staticmethod
    def reduce_double_dict_to_one(ddict, dkeys):

        ret = [{} for _ in range(len(dkeys))]

        for key, item in ddict.items():
            for i in range(len(dkeys)):
                if item is not None:
                    ret[i][key] = item[dkeys[i]]
                else:
                    ret[i][key] = -1

        return ret

    @staticmethod
    def short_num_str_to_int(num_str):

        try:
            return float(num_str)
        except:

            multiplier = num_str[-1]

            try:
                num_base = float(num_str[:-1])
            except:
                return -1

            if multiplier == 'M':
                return num_base * 1E6
            elif multiplier == 'B':
                return num_base * 1E9
            else:
                return num_base