import importlib
import pandas as pd
import BotometerRequests as br
import sqlite3 as sq
import TweetCollector
import utilities
import SqliteManager
import os
import time

Utils = utilities.Utils()
Sqm = SqliteManager.SqliteManager(path='../data/TweetData/TweetDataBase.db')


class TweetManager:
    def __init__(self):
        self.keys = ['Tweet id', 'User id', 'Screen name', 'Label', 'Search term', 'json', 'user.majority_lang',
                     'botscore',
                     'cap.english', 'cap.universal', 'raw_scores.english.astroturf', 'raw_scores.english.fake_follower',
                     'raw_scores.english.financial', 'raw_scores.english.other', 'raw_scores.english.overall',
                     'raw_scores.english.self_declared', 'raw_scores.english.spammer', 'raw_scores.universal.astroturf',
                     'raw_scores.universal.fake_follower', 'raw_scores.universal.financial',
                     'raw_scores.universal.other', 'raw_scores.universal.overall', 'raw_scores.universal.self_declared',
                     'raw_scores.universal.spammer']
        self.BR = br.BotometerRequests()
        self.path = '../data/TweetData/Tweets'

    def modify(self, to_execute: list):
        pass

    def req_tweets(self, keyword: str, num: int):
        tweets = TweetCollector.collect_tweets(phrase=keyword, history_count=num)
        bot_labels = self.BR.wrapper(from_dataframe=tweets, lite_tweet_request=True, bot_user_request=True,
                                     wanted_bot_keys=([self.keys[6]] + self.keys[8:]))
        full_df = Utils.basic_merge(tweets, bot_labels)
        full_df = Utils.order_dataframe_columns(full_df, self.keys, cut=True)

        return full_df

    def save_tweets(self, keyword: str, num: int):
        df = self.req_tweets(keyword, num)
        df.to_csv(self.path + '/' + str(time.time()) + '.csv', index=False)
        return df

    def vertical_merge(self, directory_path):
        files = os.listdir(path=directory_path)
        full_df = pd.DataFrame(columns=self.keys)
        for f in files:
            df = pd.read_csv(directory_path+'/'+f)
            full_df = pd.concat([full_df, df])

        return full_df

    def merge_and_cut(self, directory_path):
        df = self.vertical_merge(directory_path)
        for f in os.listdir(directory_path):
            os.remove(os.path.join(directory_path, f))

        df.to_csv(os.path.join(directory_path, 'Merged.csv'), index=False)


TM = TweetManager()