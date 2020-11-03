import datetime
import pandas as pd
from pandas.tseries.offsets import BDay
import random
import requests
from os import path, walk
import fnmatch
from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth


class Utils:

    @staticmethod
    def upload_file_to_gdrive(filepath, gdrive_dir=''):

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

        file_list = drive.ListFile({'q': "'root' in parents and trashed=false", 'includeTeamDriveItems': True,
                                    'supportsTeamDrives': True}).GetList()
        gfid = -1
        pgfid = -1
        for gf in file_list:
            if gf['title'] == gdrive_dir:
                gfid = gf['id']
                pgfid = gf['parents'][0]['id']

        if gfid != -1:

            if pgfid != -1:
                f = drive.CreateFile({'title': filename, 'corpora': 'teamDrive', 'includeTeamDriveItems': True,
                                      'supportsTeamDrives': True, 'teamDriveId': pgfid, 'driveId': gfid,
                                      'parents': [{'kind': 'drive#fileLink',
                                                   'teamDriveId': pgfid,
                                                   'id': gfid}]})
            else:
                f = drive.CreateFile({'title': filename, "parents": [{"kind": "drive#fileLink", "id": gfid}]})
        else:
            f = drive.CreateFile({'title': filename})

        f.SetContentFile(filepath)
        f.Upload(param={'supportsTeamDrives': True})

        f = None

    @staticmethod
    def shuffle_list(data):
        random.shuffle(data)
        return data

    @staticmethod
    def get_recent_trading_day_from_date(sel_date):
        return sel_date - datetime.timedelta(days=1) + BDay(1)

    @staticmethod
    def get_yesterday():
        return datetime.datetime.today() - datetime.timedelta(days=1)

    @staticmethod
    def get_last_trading_day():
        return Utils.get_recent_trading_day_from_date(datetime.datetime.today())

    @staticmethod
    def get_proper_date_format(date):
        return str(date).split(' ')[0]

    @staticmethod
    def get_following_week_range():

        # Current week, starting on Sunday and ending on Saturday
        today = datetime.date.today()

        start = (today + datetime.timedelta(days=7)) - \
                datetime.timedelta(days=(today + datetime.timedelta(days=1)).weekday())
        end = start + datetime.timedelta(days=6)

        return start, end

    @staticmethod
    def time_str_to_datetime(time_str):
        return datetime.datetime.strptime(time_str, '%Y/%m/%d')

    @staticmethod
    def datetime_to_time_str(ddate):
        ret = Utils.get_proper_date_format(ddate)
        return ret.replace('-', '')

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
    def find_file_pattern(pattern, root_path):

        result = []
        for root, dirs, files in walk(root_path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    result.append(path.join(root, name))

        return result

    @staticmethod
    def reduce_double_dict_to_one(ddict, dkeys):

        ret = [{} for _ in range(len(dkeys))]

        for key, item in ddict.items():
            for i in range(len(dkeys)):
                ret[i][key] = item[dkeys[i]]

        return ret
