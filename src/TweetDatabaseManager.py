import random
import pandas as pd
import BotometerRequests as br
import TweetCollector
from Scweet.scweet import scrape
from utilities import Utils
import os
import time


class TweetDatabaseManager:
    """
    Used to collect historical tweets and run botometer on them as well as multiple other useful functions
    """
    def __init__(self, use_botometer_lite=False):
        # These are the keys used in our Spam Model training database in the correct order
        # Note: modifying keys here affects Line 56 - use caution
        '''
        self.keys = ['Tweet id', 'User id', 'Screen name', 'Label', 'Search term', 'json', 'user.majority_lang',
                     'botscore',
                     'cap.english', 'cap.universal', 'raw_scores.english.astroturf', 'raw_scores.english.fake_follower',
                     'raw_scores.english.financial', 'raw_scores.english.other', 'raw_scores.english.overall',
                     'raw_scores.english.self_declared', 'raw_scores.english.spammer', 'raw_scores.universal.astroturf',
                     'raw_scores.universal.fake_follower', 'raw_scores.universal.financial',
                     'raw_scores.universal.other', 'raw_scores.universal.overall', 'raw_scores.universal.self_declared',
                     'raw_scores.universal.spammer']'''
        self.keys = ['Tweet id', 'User id', 'Screen name', 'Label', 'Search term', 'json']
        self.BR = br.BotometerRequests()

        # Path to save tweets to
        self.path = '../data/TweetData/'

        # Botometer lite config
        self.use_botometer_lite = use_botometer_lite
        if not use_botometer_lite and 'botscore' in self.keys:
            self.keys.remove('botscore')

    def modify(self, to_execute: list):
        pass

    def req_tweets(self, keyword: str, num: int):
        """
        Request a historical Twitter search and botometer scores... Uses TweetCollector.py and BotometerRequests.py

        :param keyword: Keyword to look for in Twitter
        :type keyword: str
        :param num: Number of tweets to collect (Best to do more requests of fewer tweets (100 or so at a time))
        :type num: int

        :return: Dataframe with all the keys found in self.keys
        :rtype: pd.DataFrame
        """
        tweets = TweetCollector.collect_tweets(phrase=keyword, history_count=num)

        if tweets.shape[0] == 0:
            print(f'No Tweets found for "{keyword}"...')
            return None

        else:
            print(f'{len(tweets)} Tweets collected for {keyword}...')
            bot_labels = self.BR.wrapper(from_dataframe=tweets, lite_tweet_request=self.use_botometer_lite,
                                         bot_user_request=False, wanted_bot_keys=(self.keys[6:]))
            full_df = Utils.basic_merge(tweets, bot_labels)
            full_df = Utils.order_dataframe_columns(full_df, self.keys, cut=True)

            return full_df

    @staticmethod
    def req_past_tweets(keyword: str, start_day: str, end_day: str = None, interval: int = 5):

        """Gets tweets from the past from a date. If no end_day is given, assumes up to current day.

        :param keyword: Query to search past history for.
        :type keyword: str
        :param start_day: String of start date, format of YYYYMMDD, YYYY/MM/DD, or DD-MM-YYYY
        :type start_day: str
        :param end_day: String of end date, format of YYYYMMDD, YYYY/MM/DD, or DD-MM-YYYY
        :type end_day: str
        :param interval: Day step i.e. interval = 5 means taking data every 5 days.

        :return: Dataframe with all the keys found in self.keys
        :rtype: pd.DataFrame
        """

        start = time.time()

        data = scrape(words=[keyword], hashtag=keyword, since=start_day, until=end_day, lang='en', interval=interval)

        print(time.time() - start)

        return data

    def save_tweets(self, keyword: str, num: int, filename=None):
        """
        Same as req_tweets but also saves the dataframe to a csv in the self.path directory

        :param keyword: Keyword to look for in Twitter
        :type keyword: str
        :param num: Number of tweets to collect (Best to do more requests of fewer tweets (100 or so at a time))
        :type num: int
        :param filename: Name of the file to save to (not directory),
                        otherwise saves to an auto-generated time based filename.csv
        :type filename: None or str

        :return: Dataframe with all the keys found in self.keys (also saves the dataframe to csv)
        :rtype: pd.DataFrame
        """
        df = self.req_tweets(keyword, num)
        if filename is None:
            filename = str(time.time())
        df.to_csv(self.path + filename + '.csv', index=False)
        return df

    def save_multiple_keywords(self, keywords, num: int, same_file=True, filename=None, save_to_file=False):
        """
        Same as save_tweets, but on multiple keywords with num # of tweets for each keyword

        :param keywords: List of string keywords to look for in Twitter
        :type keywords: list(str)
        :param num: Number of tweets to collect for each keyword
                    (Best to do more requests of fewer tweets (100 or so at a time))
        :type num: int
        :param same_file: Whether to save all the keyword searches to the same file or different ones
        :type same_file: bool
        :param filename: Name of the file to save to (not directory), otherwise saves to an auto-generated time based
                        filename.csv. Used only if same_file = True
        :type filename: None or str
        :param save_to_file: Whether to save to file
        :type save_to_file: bool

        :return: Dataframe of all the keywords with all the keys found in self.keys (also saves the dataframe to csv)
                or a separate csv for each keyword search
        :rtype: pd.DataFrame
        """
        if same_file:
            full_df = pd.concat([self.req_tweets(k, num) for k in keywords])
            if save_to_file:
                if filename is None:
                    filename = str(time.time())
                full_df.to_csv(self.path + filename + '.csv', index=False)
        else:
            if save_to_file:
                full_df = [self.save_tweets(k, num) for k in keywords]
            else:
                full_df = [self.req_tweets(k, num) for k in keywords]

        return full_df

    def vertical_merge(self, directory_path: str):
        """
        Takes in a directory of CSVs and concats all the dataframes into one vertically

        :param directory_path: Relative directory path (from working directory), must only contain CSVs with dataframes
                                of matching columns
        :type directory_path: str

        :return: Dataframe of all the CSVs in directory vertically combined into one
        :rtype: pd.DataFrame
        """
        files = os.listdir(path=directory_path)
        full_df = pd.DataFrame(columns=self.keys)
        for f in files:
            df = pd.read_csv(directory_path + f)
            full_df = pd.concat([full_df, df])

        return full_df

    def merge_and_cut(self, directory_path):
        """
        Takes in a directory of CSVs and concats all the dataframes into one vertically, then deletes all the files
        in directory, then saves the merged dataframe into one file named Merged.csv

        Caution: All files in directory should be CSVs with matching Dataframes (column key wise). Deletes Files!

        :param directory_path: Relative directory path (from working directory), must only contain CSVs with dataframes
                                of matching columns
        :type directory_path: str

        :return: Dataframe of all the CSVs in directory vertically combined into one, saves to directory/Merged.csv file
        :rtype: pd.DataFrame
        """
        df = self.vertical_merge(directory_path)
        for f in os.listdir(directory_path):
            os.remove(os.path.join(directory_path, f))

        print(df.value_counts('SentimentManualLabel'))

        df.to_csv(os.path.join(directory_path, 'Merged.csv'), index=False)
        return df

    @staticmethod
    def cut_skipped(df, df_path='', to_file='', inplace=False, label='Label'):
      
        if os.path.exists(df_path):
            df = pd.read_csv(df_path)

        if type(df) != pd.DataFrame:
            print('Something was wrong with the dataframe')
            return False

        df = df[df[label] >= 0]
        if to_file != '' or (inplace and df_path != ''):
            if inplace and df_path != '':
                to_file = df_path

            df.to_csv(to_file, index=False)

        return df

    def merge_no_duplicates(self, dataframes: [pd.DataFrame], directory_path=''):
        if directory_path:
            full_df = self.vertical_merge(directory_path)

        else:
            full_df = pd.concat(dataframes)

        full_df.drop_duplicates(subset=['Tweet id'], inplace=True)
        return full_df

    @staticmethod
    def sample_df(df, n, df_path='', to_base='', to_sample=''):
        if df_path:
            df = pd.read_csv(df_path)

        sample_rands = random.sample(range(len(df)), n)
        base = list(set(sample_rands) ^ set(list(range(len(df)))))
        sample = df.iloc[sample_rands]
        basedf = df.iloc[base]

        if to_base:
            basedf.to_csv(to_base, index=False)

        if to_sample:
            sample.to_csv(to_sample, index=False)

        return basedf, sample

    @staticmethod
    def restore_tweet_ids(df, df_path='', to_file='', inplace=False):
        if os.path.exists(df_path):
            df = pd.read_csv(df_path)

        if type(df) != pd.DataFrame:
            print('Something was wrong with the dataframe')
            return False

        # Tweet id is located between 55th char in json and is followed by a comma
        # We use this to restore Tweet id column by applying this method to the json column
        df = df.assign(**{'Tweet id': (lambda s: int(s['json'][55:s.find(',', 55)]))})

        if to_file != '' or (inplace and df_path != ''):
            if inplace and df_path != '':
                to_file = df_path

            df.to_csv(to_file, index=False)

        return df

    @staticmethod
    def collect_historical_tweets(start_date, end_date, queries):

        tm = TweetCollector.TwitterManager()

        for query in queries:

            filename = f'../data/TweetData/Tweets/{query}Historic{start_date.replace("-", "")}-{end_date.replace("-", "")}.csv'
            out_filename = f'../data/TweetData/Tweets/{query}{start_date.replace("-", "")}-{end_date.replace("-", "")}.csv'

            if not os.path.exists(filename):
                tweets = TweetDatabaseManager.req_past_tweets(query, start_date, end_date, interval=1)

                Utils.write_dataframe_to_csv(tweets, filename, write_index=False)

            if os.path.exists(filename):
                df = tm.tweet_urls_to_dataframe(filename, query)

                Utils.write_dataframe_to_csv(df, out_filename, write_index=False)


if __name__ == '__main__':

    queries = ['DPZ', 'TTD', '$DPZ', '$TTD', "Domino's", 'Dominos', 'Trade Desk']

    tdm = TweetDatabaseManager()
    tweets = tdm.save_multiple_keywords(keywords=queries, num=1000, same_file=False, filename='tweets',
                                        save_to_file=True)
