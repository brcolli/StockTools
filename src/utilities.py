import datetime
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.holiday import GoodFriday
from pandas.tseries.offsets import CustomBusinessDay
import random
import requests
from os import path, walk, makedirs
import fnmatch
from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth
from io import StringIO
import csv
import re
import itertools
import ast
import matplotlib.pyplot as plt
import math


"""utilities

Description:
Module for functions that can be reused and shared between various other modules, contains a vast array of tools.

Authors: Benjamin Collins
Date: April 22, 2021 
"""


class Utils:

    # Define business day
    USFederalHolidayCalendar.rules.pop(7)  # Allow trading on Veterans Day
    USFederalHolidayCalendar.rules.pop(6)  # Allow trading on Columbus Day
    USFederalHolidayCalendar.rules.append(GoodFriday)  # Do not allow trading on Good Friday
    BDay = CustomBusinessDay(calendar=USFederalHolidayCalendar())

    @staticmethod
    def upload_files_to_gdrive(filepaths, gdrive_dir=''):

        if not filepaths:
            return

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

        # WARNING: This assumes that the file list does not change during the for loop!
        # Only an issue for first level directories
        file_list_root = drive.ListFile(
            {'q': "trashed=false", 'includeItemsFromAllDrives': True,
             'supportsAllDrives': True, 'corpora': 'allDrives'}).GetList()

        for filepath in filepaths:

            if not path.exists(filepath):
                continue

            filename = filepath.split('/')[-1]
            dirs = gdrive_dir.split('/')
            dirs = [x for x in dirs if x]

            # folder_id = '12Tcsl57iFVvQyCPr49-VzUfcbHcrXU34'
            # file_list = drive.ListFile({'q': f"parents in '{folder_id}' and trashed=false",
            # 'includeItemsFromAllDrives':
            # True,
            # 'supportsAllDrives': True, 'corpora': 'allDrives'}).GetList()
            folder_id = ''
            curr_dir_id = 0
            gf_id = 0
            file_list = file_list_root
            while gf_id < len(file_list):

                gf = file_list[gf_id]
                gf_id += 1

                if gf['title'] == dirs[curr_dir_id] and gf['shared'] is True:

                    folder_id = gf['id']  # Found latest sub directory

                    # Check if there are more subdirectories to iterate on
                    curr_dir_id += 1
                    if len(dirs) == curr_dir_id:
                        # Have found the last sub directory, exit
                        break
                    else:
                        # Update file_list
                        file_list = drive.ListFile({'q': f"parents in '{folder_id}' and trashed=false",
                                                    'includeItemsFromAllDrives': True, 'supportsAllDrives': True,
                                                    'corpora': 'allDrives'}).GetList()
                        gf_id = 0

            # If it did not get through the entire sub directory list, must create the rest of the sub dirs
            if len(dirs) > curr_dir_id:

                while curr_dir_id < len(dirs):

                    # Create drive sub folder, add parents
                    sub_dir = drive.CreateFile({'title': dirs[curr_dir_id],
                                                'mimeType': 'application/vnd.google-apps.folder',
                                                'parents': [{'id': folder_id}]})
                    sub_dir.Upload()
                    folder_id = sub_dir['id']

                    curr_dir_id += 1

            if folder_id != '':

                # If file exists, skip
                file_list = drive.ListFile({'q': f"parents in '{folder_id}' and trashed=false",
                                            'includeItemsFromAllDrives': True, 'supportsAllDrives': True,
                                            'corpora': 'allDrives'}).GetList()
                duplicate = False
                for gf in file_list:
                    if gf['title'] == filename:
                        duplicate = True
                if duplicate:
                    continue

                f = drive.CreateFile({'title': filename, 'corpora': 'allDrives', 'includeItemsFromAllDrives': True,
                                      'supportsAllDrives': True,
                                      'parents': [{'kind': 'drive#fileLink',
                                                   'id': folder_id}]})
            else:
                f = drive.CreateFile({'title': filename})

            f.SetContentFile(filepath)
            f.Upload(param={'supportsAllDrives': True})  # TODO handle BrokenPipeError: [Errno 32]

    # Reformatting short interest file to a proper csv
    @staticmethod
    def replace_line_to_comma(text):
        text = text.replace(',', '/')
        return text.replace('|', ',')

    @staticmethod
    def regsho_txt_to_df(text, vol_lim=-1):

        # Convert text into a dataframe
        sio = StringIO(text).__str__()
        df = pd.read_csv(sio, sep=',')[:-1]
        df = df[df['TotalVolume'] >= vol_lim]  # Only take rows with volumes greater than filter

        return df

    # Use an arbitrary day to get finra's ticker list for that day
    @staticmethod
    def get_all_tickers(day=None):

        if not day:
            day = Utils.get_last_trading_day()

        url = 'http://regsho.finra.org'
        short_file_prefix = 'CNMSshvol'

        if type(day) is not str:
            day = Utils.datetime_to_time_str(day)

        # Import finra csv
        filename = short_file_prefix + day
        data = Utils.get_file_from_url(url, filename + '.txt')
        text = Utils.replace_line_to_comma(data)
        df = Utils.regsho_txt_to_df(text)

        return df['Symbol'].tolist()

    @staticmethod
    def shuffle_data(data, seed=123):

        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            data = data.sample(frac=1).reset_index(drop=True)
            return data
        else:
            random.seed(seed)
            random.shuffle(data)
            return data

    @staticmethod
    def sample_data(data, sample_size):
        return random.sample(data, sample_size)

    @staticmethod
    def get_previous_trading_day_from_date(sel_date=datetime.datetime.today()):
        return sel_date - Utils.BDay

    @staticmethod
    def get_yesterday():
        return datetime.datetime.today() - datetime.timedelta(days=1)

    @staticmethod
    def get_last_trading_day():
        return datetime.datetime.today() + datetime.timedelta(days=1) - Utils.BDay

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
    def get_week_range(dday):

        # Current week, starting on Sunday and ending on Saturday
        start = dday - datetime.timedelta(days=(dday + datetime.timedelta(days=1)).weekday())
        end = start + datetime.timedelta(days=6)

        return start, end

    @staticmethod
    def get_following_week_range():
        return Utils.get_week_range((datetime.date.today() + datetime.timedelta(days=7)))

    @staticmethod
    def reformat_time_str(time_str):

        if '-' in time_str:
            return datetime.datetime.strptime(time_str, "%d-%m-%Y").strftime("%Y%m%d")

        return datetime.datetime.strptime(time_str, "%m/%d/%Y").strftime("%Y%m%d")

    @staticmethod
    def time_str_to_datetime(time_str):

        if time_str == '':
            return None

        if '/' in time_str:
            return datetime.datetime.strptime(time_str, '%Y/%m/%d')
        elif '-' in time_str:
            return datetime.datetime.strptime(time_str, '%d-%m-%Y')
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

    @staticmethod
    def epoch_to_time_str(epoch):
        return Utils.datetime_to_time_str(Utils.epoch_to_datetime(epoch))

    @staticmethod
    def write_list_of_dicts_to_csv(data, filename):

        if not data:
            print('Data is empty. Please try again.')
        else:
            print('Writing data to ' + filename + '...')

            # Attempt to write to csv
            try:

                with open(filename, 'w', newline='') as f:

                    header = list(data[0].keys())
                    writer = csv.DictWriter(f, fieldnames=header)

                    writer.writeheader()
                    for row in data:
                        writer.writerow(row)
            except:
                print('Could not open ' + filename + '. Is the file open?')

    '''
    Writes a pandas DataFrame to a csv
    '''
    @staticmethod
    def write_dataframe_to_csv(df, filename, write_index=True):

        if df.empty:
            print('Data is empty. Please try again.')
        else:
            print('Writing data to ' + filename + '...')

            # Attempt to write to csv
            try:
                df.to_csv(filename, index=write_index)
                return True
            except:
                print('Could not open ' + filename + '. Is the file open?')
                return False

    @staticmethod
    def order_dataframe_columns(df, keys, cut=True):
        if not cut:
            diff = set(keys) ^ set(df.columns.tolist())
            keys = keys + list(diff)

        return df[keys]

    @staticmethod
    def min_of_columns(df_results: pd.DataFrame, columns):
        """
        Returns a column of minimums from a dataframe and a list of keys.

        :param df_results: pandas dataframe
        :type df_results: pandas.DataFrame

        :param columns: list of column names to compare
        :type columns: list(str)

        :return: A column of row-wise minimums for the given keys
        :rtype: pandas Dataframe column
        """
        return df_results[columns].min(axis=1)

    @staticmethod
    def max_of_columns(df_results: pd.DataFrame, columns):
        """
        Returns a column of maximums from a dataframe and a list of keys.

        :param df_results: pandas dataframe
        :type df_results: pandas.DataFrame

        :param columns: list of column names to compare
        :type columns: list(str)

        :return: A column of row-wise maximums for the given keys
        :rtype: pandas Dataframe column
        """
        return df_results[columns].max(axis=1)

    @staticmethod
    def segment_list(lst, n):
        """
        Segments a list into lists of size n with the last list being the smallest if len(lst)%n != 0

        :param lst: list to segment
        :type lst: list

        :param n: number of elements per list
        :type n: int

        :return: segmented list
        :rtype: list(list, list, list)
        """
        return [lst[i:i + n] for i in range(0, len(lst), n)]

    @staticmethod
    def flatten(lst):
        """
        Flattens a list of lists into one list
        :param lst: list of lists to flatten
        :type lst: list

        :return: flattened list
        :rtype: list
        """
        return list(itertools.chain.from_iterable(lst))

    @staticmethod
    def strings_to_dicts(strings):
        """
        Converts a list of json style strings to a list of dictionaries.

        :param strings: list of strings
        :type strings: list(str)

        :return: list of dicts
        :rtype: list(dict)

        """
        return [ast.literal_eval(s) for s in strings]

    def merge_dataframes(self, original, new, original_id_key, new_id_key, new_data_keys):
        """
        Horizontally merges two dataframes with replacement of missing values in the new according to an id column in
        both dataframes.

        :param original: original dataframe (must include in it all of the ids of the new dataframe)
        :type original: pandas.DataFrame

        :param new: new dataframe (may have an equal amount or less rows than original, but cannot have ids not in the
                                    original)
        :type new: pandas.DataFrame

        :param original_id_key: key of the id column in the original df
        :type original_id_key: str

        :param new_id_key: key of the id column in the new df
        :type new_id_key: str

        :param new_data_keys: list of data keys in the new df (to merge onto original)
        :type new_data_keys: list(str)

        :return: Merged dataframe
        :rtype: pandas.DataFrame
        """
        if original.shape[0] != new.shape[0]:
            insertion_indices = self.find_missed(original[original_id_key].tolist(), new[new_id_key].tolist())
            for key in new_data_keys:
                original[key] = self.insert_at_indices(new[key].tolist(), insertion_indices, -1)
        else:
            for key in new_data_keys:
                original[key] = new[key]

        return original

    @staticmethod
    def basic_merge(original, new):
        """
        Horizontally merges two pandas dataframe of equal size

        :param original: original df
        :type original: pd.DataFrame

        :param new: new df
        :type new: pd.DataFrame

        :return: merged df
        :rtype: pandas.DataFrame
        """

        duplicates = list(set(original.keys()) & set(new.keys()))
        new = new.drop(labels=duplicates, axis=1)
        return pd.concat([original, new], axis=1)

    @staticmethod
    def find_missed(expected, result):
        """
        Returns a list of missed indices between an ordered expected and result list.

        :param expected: expected data
        :type expected: list

        :param result: the result data with missing values
        :type result: list

        :return: list of indices to reinsert missed values
        :rtype: list(int)
        """
        if len(expected) == len(result):
            return []
        my = []
        x = 0
        for i in range(len(expected)):
            try:
                if result[i - x] != expected[i]:
                    my.append(i)
                    x += 1
            except IndexError:
                my.append(i)

        return my

    @staticmethod
    def insert_at_indices(data, indices, null=-1):
        """
        Inserts some null value into a list of data at the appropriate indices.

        :param data: data with missed values
        :type data: list

        :param indices: list of indices to insert at, from self.find_missed method
        :type indices: list(int)

        :param null: null value to insert at missing places
        :type null: any

        :return: data with nulls inserted for missing values
        :rtype: list
        """
        for i in indices:
            data.insert(i, null)
        return data

    @staticmethod
    def load_csv_to_dataframe(filename):
        return pd.read_csv(filename, index_col=[0])

    @staticmethod
    def get_full_path_from_file_date(file_date, file_prefix='', file_suffix='', root='../data/', do_daily=False):

        dd = Utils.time_str_to_datetime(file_date)

        # Get the weekly range
        start, end = Utils.get_week_range(dd)

        # If week range is split between years, pick the earlier year
        file_year = start.year

        # Check if year directory exists, and create if not
        full_path = root + str(file_year) + '/'
        if not path.exists(full_path):
            makedirs(full_path)

        # Add 0s to the front of months less than 10
        if start.month < 10:
            sm = '0' + str(start.month)
        else:
            sm = str(start.month)
        if end.month < 10:
            em = '0' + str(end.month)
        else:
            em = str(end.month)

        # Add 0s to the front of days less than 10
        if start.day < 10:
            sd = '0' + str(start.day)
        else:
            sd = str(start.day)
        if end.day < 10:
            ed = '0' + str(end.day)
        else:
            ed = str(end.day)

        # Check if weekly range directory exists, and create if not
        file_week = sm + sd + '-' + em + ed
        full_path += file_week + '/'
        if not path.exists(full_path):
            makedirs(full_path)

        # Split weekly directory into day directories
        if do_daily:

            full_path += file_date + '/'
            if not path.exists(full_path):
                makedirs(full_path)

        # Combine full path with filename
        full_path += file_prefix + file_date + file_suffix

        return full_path

    @staticmethod
    def get_tickers_from_csv(filename):

        with open(filename) as f:
            reader = csv.reader(f)
            data = list(reader)

        return [j for i in data for j in i]

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

    @staticmethod
    def get_matches_from_regex(text, pattern):
        m = re.search(pattern, text)
        if m:
            return m.groups()
        else:
            return ()

    @staticmethod
    def clean_json_bad_str_char(json_data):

        p = re.compile('(?<!\\\\)\'')
        clean = p.sub('\"', json_data)
        clean = clean.replace("\\'s", "\'s")

        return clean

    @staticmethod
    def safe_str_to_dict(string: str or dict) -> dict:
        """
        If you are unsure whether you are working with a dictionary or string representation of dictionary, returns a
        dictionary

        :param string: Either string (json) dictionary or dictionary (generally a json of a stored tweet)
        :type string: str or dict

        :return: The dictionary version of the string (or dictionary itself)
        :rtype: dict
        """
        if type(string) == str:
            string = eval(string)
        return string

    @staticmethod
    def parse_json_tweet_data_from_csv(filename, json_headers):
        df = pd.read_csv(filename)
        return Utils.parse_json_tweet_data(df, json_headers)

    @staticmethod
    def parse_json_tweet_data(df, json_headers):

        json_dict = {}
        for header in json_headers:
            json_dict[header] = []

        if 'json' not in df.columns:
            return df

        for row, col in df.iterrows():

            json_data = Utils.safe_str_to_dict(col['json'])
            for header in json_headers:

                if header == 'full_text' and 'full_text' not in json_data.keys():
                    json_dict[header].append(json_data['text'])
                else:
                    if header in json_data.keys():
                        json_dict[header].append(json_data[header])

        for header in json_headers:
            if header not in df.columns and json_dict[header]:
                df[header] = json_dict[header]

        return df

    @staticmethod
    def calculate_ml_measures(results, labels):
        """
        Calculates the accuracy, precision, recall, f1 score, mcor as well as tp,fp,tn,fn from a list of result labels
        and actual labels. Skips -1 results and labels.

        :param results: list of result labels
        :type results: list(int)

        :param labels: list of actual labels
        :type labels: list(int)

        :return: accuracy, precision, recall, f1 score, mcor, (total, true positives, false positives, true negatives,
                                                        false negatives)
        :rtype: float, float, float, float, float, (int, int, int, int, int)
        """

        tp = 0
        fp = 0
        tn = 0
        fn = 0
        sk = 0
        for i in range(len(results)):
            r = results[i]
            e = labels[i]
            if r == -1 or e == -1:
                sk += 1

            elif e == 0:
                if r == 0:
                    tn += 1
                elif r == 1:
                    fp += 1

            elif e == 1:
                if r == 0:
                    fn += 1
                elif r == 1:
                    tp += 1

        accuracy = (tp + tn) / max((tp + fp + fn + tn), 1)
        precision = tp / max((tp + fp), 1)
        recall = tp / max((tp + fn), 1)
        f1 = 2 * (recall * precision) / max((recall + precision), 1)

        eps = 1.0
        while eps + 1 > 1:
            eps /= 2
        eps *= 2

        mcor_numerator = (tp * tn - fp * fn)
        mcor_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcor = mcor_numerator / (mcor_denominator + eps)

        return accuracy, precision, recall, f1, mcor, (tp+fp+tn+fn, tp, fp, tn, fn)

    @staticmethod
    def normalize(m: float, rmin: float, rmax: float, tmin: float, tmax: float) -> float:

        """Normalizes a number into a given range.

        :param m: Number to normalize
        :type m: int
        :param rmin: Lower bound of range of number m
        :type rmin: int
        :param rmax: Upper bound of range of number m
        :type rmax: int
        :param tmin: Lower bound of desired range to normalize to
        :type tmin: int
        :param tmax: Upper bound of desired range to normalize to
        :type tmax: int

        :return: Number normalized to be within range tmin <= ret <= tmax
        :rtype: float
        """

        return round(((m - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin, 1)

    @staticmethod
    def posnorm(m, rmin, rmax):
        return Utils.normalize(m, rmin, rmax, 65.1, 100)

    @staticmethod
    def neunorm(m, rmin, rmax, nmin, nmax):
        return Utils.normalize(m, rmin, rmax, nmin, nmax)

    @staticmethod
    def negnorm(m, rmin, rmax):
        return Utils.normalize(m, rmin, rmax, 0, 34.9)

    @staticmethod
    def tweet_data_analysis(df, df_path='', plot=False):
        if path.exists(df_path):
            df = pd.read_csv(df_path)

        if type(df) != pd.DataFrame:
            print('Something was wrong with the dataframe')
            return False

        values = df['Label'].value_counts()
        labels = {0: 'Clean', 1: 'Spam', -1: 'Skipped'}
        res_dict = {}
        for k, v in labels.items():
            try:
                x = values[k]
            except KeyError:
                x = 0
            res_dict[v] = {}
            res_dict[v]['Count'] = x

        res_dict['Total'] = {}
        res_dict['Total']['Count'] = len(df)
        res_dict['Total']['Percent'] = 1
        for k in res_dict.keys():
            if k != 'Total':
                res_dict[k]['Percent'] = res_dict[k]['Count'] / res_dict['Total']['Count']

        if plot:
            plt.figure()
            plt.bar(list(res_dict.keys()), [v['Count'] for v in res_dict.values()])
            plt.show()

        return res_dict

    @staticmethod
    def compare_tweet_label(a, b):
        if b == -1:
            if a == -1:
                return -1
            else:
                return -2
        elif a == b:
            return 0
        else:
            if b == 1:
                return 1
            else:
                return 2

    @staticmethod
    def compare_many_tweet_labels(original, new):
        return [Utils.compare_tweet_label(a, b) for a, b in zip(original, new)]
