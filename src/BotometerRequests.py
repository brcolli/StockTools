import botometer
import numpy as np
import pandas
import pandas as pd
import importlib
import os
from dotenv import load_dotenv
from enum import Enum
import time
import itertools
import tweepy
import ast
import json
Utils = importlib.import_module('utilities').Utils

# TODO Add wrapper functions for main entry points
# TODO Add functions for accuracy, recall, precision, F-scores
# TODO Add doc string comments for all methods, class, and the overall module

class RequestCase(Enum):
    ALL = 1
    LITE_U = 2

class BotometerRequests:

    def __init__(self, threshold=0.5, classifier='raw_scores.english.overall', cap_classification=False,
                 best_language=False, twitter_api=False):
        load_dotenv('../doc/BotometerKeys.env')

        self.rapidapi_key = os.getenv('rapidapi_key')
        self.twitter_app_auth = {
            'consumer_key': os.getenv('consumer_key'),
            'consumer_secret': os.getenv('consumer_secret')
            }
        self.ALL_BOT_KEYS = os.getenv('ALL_BOT_KEYS').split(',')
        self.ALL_LITE_KEYS = ['user_id', 'tweet_id', 'botscore']

        self.bot_null_result = ast.literal_eval(os.getenv('null_bot_dictionary'))
        self.lite_null_result = {"botscore": -1, "tweet_id": -1, "user_id": -1}
        self.u_lite_null_result = {"u_botscore": -1, "tweet_id": -1, "user_id": -1}

        self.threshold = threshold
        self.best_language = best_language
        self.auth = False
        self.twitter_api = twitter_api

        if best_language and cap_classification:
            self.classifier = ['cap.english', 'cap.universal']
        elif best_language and not cap_classification:
            self.classifier = [classifier, self.replace_english_with_universal(classifier)]
        elif not best_language and cap_classification:
            self.classifier = ['cap.english']
        else:
            self.classifier = [classifier]

        self.bom = botometer.Botometer(wait_on_ratelimit=True,
                                  rapidapi_key=self.rapidapi_key,
                                  **self.twitter_app_auth)

        self.lite = botometer.BotometerLite(rapidapi_key=self.rapidapi_key, **self.twitter_app_auth)

    def setup_tweepy_api(self):
        self.auth = tweepy.OAuthHandler(self.twitter_app_auth['consumer_key'], self.twitter_app_auth['consumer_secret'])
        self.auth.set_access_token(os.getenv('shToken'), os.getenv('scToken'))
        self.twitter_api = tweepy.API(self.auth, wait_on_rate_limit=True)

    @staticmethod
    def replace_english_with_universal(key):
        a = key.find("english")
        if a == -1:
            return key
        b = a + len("english")
        return key[:a] + 'universal' + key[b:]

    def next_result_or_blank(self, result):
        if 'error' == next(iter(result[1])):
            return self.bot_null_result
        else:
            return result[1]

    def request_botometer_results_map(self, user_ids):
        """Requests botometer scores and returns a map of dictionaries with the provided scores, errors replaced with -1

                        :param user_ids: list of Twitter user ids
                        :type user_ids: list(str)

                        :return: map of dictionaries of results from botometer, error replaced with blanks
                        :rtype: map(dict)
                        """
        while True:
            try:
                bom_generator = self.bom.check_accounts_in(user_ids)
                break
            except:
                print('Some error with botometer... Trying again in 1 second')
                time.sleep(1)
        return map(self.next_result_or_blank, bom_generator)

    def request_botometer_results(self, user_ids):
        """Requests botometer scores and returns a list of dictionaries with the provided scores, cleaned for errors

                :param user_ids: list of Twitter user ids
                :type user_ids: list(str)

                :return: list of dictionaries of results from botometer, without error values
                :rtype: list(dict)
                """

        while True:
            results = []
            error_count = 0
            try:
                for _, res in self.bom.check_accounts_in(user_ids):
                    if 'error' != next(iter(res)):
                        results.append(res)
                    else:
                        results.append(self.bot_null_result)
                        error_count += 1
                break
            except:
                print('Some error with botometer... Trying again in 1 second')
                time.sleep(1)

        print(f'Unable to score {error_count} users... {len(results)-error_count} users scored')
        return results

    @staticmethod
    def clean_errors(results, err_str='Error scores removed: '):
        """Removes error instances from a list of botometer results.

                :param results: list of botometer result dictionaries
                :type results: list(dict)

                :param err_str: string to print out before the error count
                :type err_str: str

                :return: list of dictionaries of botometer results, without error values
                :rtype: list(dict)
                """
        cleaned = [res for res in results if 'error' != next(iter(res))]
        print(f'{err_str}{len(results)-len(cleaned)}')
        return cleaned

    @staticmethod
    def replace_errors(results, null_result):
        """Replaces error instances from a list of botometer results with null dictionary skeleton

                :param results: list of botometer result dictionaries
                :type results: list(dict)

                :return: list of dicts of botometer results, without error values as null dicts
                :rtype: list(dict)
                """
        return [res if 'error' != next(iter(res)) else null_result for res in results]

    def get_unwanted_keys(self, wanted_keys):
        """Takes a list of wanted botometer keys and subtracts it from the list of all botometer keys to yield unwanted

                :param wanted_keys: list of keys to store in dataframe
                :type wanted_keys: list(str)

                :return: list of unwanted keys
                :rtype: list(str)
                """
        return list(set(wanted_keys) ^ set(self.ALL_BOT_KEYS))

    def results_to_dataframe(self, results, wanted_keys):
        """Takes a list of botometer results and converts them to a pandas dataframe keeping only ids and wanted keys

                :param results: list of botometer result dictionaries
                :type user_ids: list(dict)

                :param wanted_keys: list of keys to store in dataframe
                :type wanted_keys: list(str)

                :return: pandas dataframe with column id and wanted keys columns
                :rtype: pandas dataframe size: [1 + wanted_keys x len(results)]
                """

        all_keys = pd.json_normalize(results)
        ids = all_keys['user.user_data.id_str']
        dropped = all_keys.drop(labels=self.get_unwanted_keys(wanted_keys), axis=1)
        dropped.insert(0, "user_id", ids)
        return dropped

    @staticmethod
    def min_of_columns(df_results: pandas.DataFrame, columns):
        return df_results[columns].min(axis=1)

    @staticmethod
    def max_of_columns(df_results: pandas.DataFrame, columns):
        return df_results[columns].max(axis=1)

    @staticmethod
    def classify_results(data, threshold):
        """Takes a list of values and returns a label list according to the threshold

                :param data: list of values, -1 for null or score
                :type user_ids: list(float)

                :param threshold: decimal value in [0, 1] with score > threshold indicating a bot
                :type threshold: float

                :return: list of labels the length of the data: -1 for no label, 0 for clean, 1 for spam
                :rtype: list(int)
                """

        return [-1 if x == -1 else 0 if x <= threshold else 1 for x in data]

    @staticmethod
    def thresholds_x_classifiers(thresholds, classifiers):
        new_thresholds = [t for t in thresholds for c in classifiers]
        new_classifiers = [c for t in thresholds for c in classifiers]
        names = [new_classifiers[i] + '.' + str(new_thresholds[i]) for i in range(len(new_classifiers))]
        return new_thresholds, new_classifiers, names

    def classify_one_to_one(self, df_results, thresholds, classifiers, names):
        labels = {n: self.classify_results(df_results[c], t) for c, t, n in zip(classifiers, thresholds, names)}
        return pd.DataFrame(labels)

    def classify_for_each_threshold(self, df_results, thresholds, classifiers):
        new_thresholds, new_classifiers, names = self.thresholds_x_classifiers(thresholds, classifiers)
        return self.classify_one_to_one(df_results, new_thresholds, new_classifiers, names)

    @staticmethod
    def segment_list(lst, n):
        return [lst[i:i + n] for i in range(0, len(lst), n)]

    @staticmethod
    def flatten(lst):
        return list(itertools.chain.from_iterable(lst))

    def request_botometer_lite_tweet_results_from_objects(self, tweets):
        """Takes a list of tweet Tweepy objects and returns a list of botometer lite results, cleaned for errors

                :param tweets: list of tweet objects
                :type tweets: list(tweepy Tweet Object)

                :return: list of botometer results
                :rtype: list(dict)
                """

        if type(tweets[0]) == str:
            tweets = self.strings_to_dicts(tweets)
        if len(tweets) > 100:
            scores = self.flatten([self.lite.check_accounts_from_tweets(x) for x in self.segment_list(tweets, 100)])
        else:
            scores = self.lite.check_accounts_from_tweets(tweets)

        return self.replace_errors(scores, self.lite_null_result)

    def get_tweets_from_ids(self, tweet_ids, api):
        """Takes a list of tweet ids and Tweepy api and returns a list of json tweet objects.

                    :param tweet_ids: list of tweet ids
                    :type tweets: list(str)

                    :param api: Initialized tweepy API, use self.setup_tweepy_api and self.twitter_api or external api
                    :type api: tweepy.API

                    :return: list of json tweet objects
                    :rtype: list(json)
                    """

        tweets = []
        error_count = 0
        for tid in tweet_ids:
            try:
                tweets.append(api.get_status(tid)._json)
                print(f'Tweet number: {len(tweets) + error_count}')
            except:
                print(f'Tweet number: {len(tweets) + error_count} error')
                error_count += 1
        print(f'Unable to fetch {error_count} tweets... fetched {len(tweets)} tweets')

        return tweets

    def request_botometer_lite_user_results(self, user_ids):
        """Takes a list of user ids and returns a list of botometer lite results for those users, cleaned for errors

                    :param user_ids: list of user ids
                    :type user_ids: list(str)

                    :return: list of botometer results
                    :rtype: list(dict)
                    """

        if len(user_ids) > 100:
            scores = self.flatten([self.lite.check_accounts_from_user_ids(x) for x in self.segment_list(user_ids, 100)])

        else:
            scores = self.lite.check_accounts_from_user_ids(user_ids)

        return self.replace_errors(scores, self.lite_null_result)

    def lite_results_to_dataframe(self, results, keep_tweet_ids=True):
        """Takes a list of botometer lite results and returns a pandas dataframe

                    :param results: list of botometer lite results
                    :type results: list(str)

                    :param keep_tweet_ids: whether or not to keep the tweet_ids
                    :type keep_tweet_ids: bool

                    :return: pandas dataframe with user_id, tweet_id, botscore columns
                    :rtype: pandas dataframe
                    """

        df = pd.json_normalize(results)
        df = df[self.ALL_LITE_KEYS]

        if None == df['tweet_id'][0]:
            df = df.rename(columns={'botscore': 'u_botscore'})

        if not keep_tweet_ids:
            df = df.drop(labels=['tweet_id'], axis=1)
        return df

    @staticmethod
    def strings_to_dicts(strings):
        return [ast.literal_eval(s) for s in strings]

    def merge_dataframes(self, original, new, original_id_key, new_id_key, new_data_keys):
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
        return pd.concat([original, new], axis=1)

    @staticmethod
    def find_missed(expected, result):
        if len(expected) == len(result):
            return []
        my = []
        x = 0
        for i in range(len(expected)):
            if result[i - x] != expected[i]:
                my.append(i)
                x += 1

        return my

    @staticmethod
    def insert_at_indices(data, indices, null):
        for i in indices:
            data.insert(i, null)
        return data

    @staticmethod
    def replace_missed(expected, result_indices, results, null_result):
        if len(expected) == len(result_indices):
            return results
        for i in range(len(expected)):
            if result_indices[i] != expected[i]:
                result_indices.insert(i, expected[i])
                results.insert(i, null_result)

        return results

    @staticmethod
    def calculate_precision(results, labels):
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

        accuracy = (tp+tn)/(tp+fp+fn+tn-sk)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = 2 * (recall * precision) / (recall + precision)

        return accuracy, precision, recall, f1

    @staticmethod
    def calc_averages(results, labels):
        spam_sum = 0
        clean_sum = 0
        spam_num = 0
        clean_num = 0
        for i in range(len(results)):
            if results[i] == -1 or labels[i] == -1:
                pass
            elif labels[i] == 0:
                clean_sum += results[i]
                clean_num += 1
            elif labels[i] == 1:
                spam_sum += results[i]
                spam_num += 1

        return clean_sum/clean_num, spam_sum/spam_num

    def calc_multi_averages(self, df, result_keys, label_key):
        averages = []
        for key in result_keys:
            averages.append(self.calc_averages(df[key], df[label_key]))

        return averages

    def wrapper(self, tweet_ids=None, tweet_objects=None, user_ids=None, bot_user_request=False,
                lite_user_request=False, lite_tweet_request=False, thresholds=None, classifiers=None,
                existing_api=None, wanted_bot_keys=None):

        if user_ids is None:
            bot_user_request = False
            lite_user_request = False

        if tweet_objects is None:
            if tweet_ids is None:
                lite_tweet_request = False
            elif lite_tweet_request:
                if existing_api is None:
                    self.setup_tweepy_api()
                    api = self.twitter_api
                else:
                    api = existing_api
                print("Collecting tweet jsons from ids... Might take a while")
                tweet_objects = self.get_tweets_from_ids(tweet_ids, api)
        elif tweet_ids is None:
            tweet_ids = [obj['id'] for obj in tweet_objects]

        if wanted_bot_keys is None:
            wanted_bot_keys = ['cap.english']

        if thresholds is None or classifiers is None:
            classify = False
        else:
            classify = True

        result_df = pd.DataFrame()

        if lite_tweet_request:
            lite_results = self.request_botometer_lite_tweet_results_from_objects(tweet_objects)
            temp_df = self.lite_results_to_dataframe(lite_results)
            result_df['Tweet id'] = tweet_ids
            result_df = self.merge_dataframes(result_df, temp_df, 'Tweet id', 'tweet_id', ['botscore'])

        if lite_user_request:
            lite_user_results = self.request_botometer_lite_user_results(user_ids)
            temp_df = self.lite_results_to_dataframe(lite_user_results)
            result_df['User id'] = user_ids
            result_df = self.merge_dataframes(result_df, temp_df, 'User id', 'user_id', ['u_botscore'])

        if bot_user_request:
            bot_user_results = self.request_botometer_results_map(user_ids)
            temp_df = self.results_to_dataframe(bot_user_results, wanted_bot_keys)
            temp_df = temp_df.drop('user_id', axis=1)
            result_df = self.basic_merge(result_df, temp_df)

        if classify:
            label_df = self.classify_for_each_threshold(result_df, thresholds, classifiers)
            result_df = self.basic_merge(result_df, label_df)

        return result_df
