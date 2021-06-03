import botometer
import numpy as np
import pandas as pd
import importlib
import os
from dotenv import load_dotenv
import time
import itertools
import tweepy
import ast
import json
Utils = importlib.import_module('utilities').Utils

# TODO Add wrapper functions for main entry points
# TODO Add functions for accuracy, recall, precision, F-scores
# TODO Add doc string comments for all methods, class, and the overall module

class BotometerRequests:

    def __init__(self, threshold=0.5, classifier='raw_scores.english.overall', cap_classification=False,
                 best_language=False, tweet_api=False):
        load_dotenv('../doc/BotometerKeys.env')

        self.rapidapi_key = os.getenv('rapidapi_key')
        self.twitter_app_auth = {
            'consumer_key': os.getenv('consumer_key'),
            'consumer_secret': os.getenv('consumer_secret')
            }
        self.ALL_BOT_KEYS = os.getenv('ALL_BOT_KEYS').split(',')
        self.ALL_LITE_KEYS = ['user_id', 'tweet_id', 'botscore']

        self.bot_null_result = ast.literal_eval(os.getenv('null_bot_dictionary'))
        self.lite_null_result = {"botscore": None, "tweet_id": None, "user_id": None}

        self.threshold = threshold
        self.best_language = best_language
        self.tweet_api = tweet_api

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

    def classify_results(self, df_results, threshold=0.5, classifier='raw_scores.english.overall',
                         cap_classification=False, botscore=False, best_language=False):
        """Takes a dataframe of botometer results and adds spam/clean label depending on threshold and classifier

                :param df_results: dataframe of botometer results, must have classifier as a column
                :type user_ids: pandas dataframe

                :param threshold: decimal value in [0, 1] with score > threshold indicating a bot
                :type threshold: float

                :param classifier: key of the classifier to be used, must be a column in df_results
                :type classifier: str

                :param cap_classification: whether or not to use the cap score for classification,
                                            overrides the classifier if True
                :type cap_classification: bool

                :param botscore: whether or not to use the botscore for classification
                :type botscore: bool

                :param best_language: whether or not to use the lower of universal and english scores
                :type best_language: bool

                :return: df_results with an added column 'label' which has binary classification
                :rtype: pandas dataframe
                """

        if cap_classification:
            classifier = 'cap.english'

        if botscore:
            classifier = 'botscore'

        if best_language and not botscore:
            compare_col = df_results[[classifier, self.replace_english_with_universal(classifier)]].min(axis=1)

        else:
            compare_col = df_results[classifier]

        labels = np.where(compare_col >= threshold, 0, 1)
        df_results['bot_label'] = labels
        return df_results

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

        if len(tweets) > 100:
            scores = self.flatten([self.lite.check_accounts_from_tweets(x) for x in self.segment_list(tweets, 100)])

        else:
            scores = self.lite.check_accounts_from_tweets(tweets)

        return self.replace_errors(scores, self.lite_null_result)

    def request_botometer_lite_tweet_results_from_ids(self, tweet_ids, api):
        """Takes a list of tweet ids and turns them into tweet objects then returns a list of botometer lite results,
            cleaned for errors

                    :param tweet_ids: list of tweet ids
                    :type tweets: list(str)

                    :param api: Initialized tweepy API, should be done outside of this class
                    :type api: tweepy.API

                    :return: list of botometer results
                    :rtype: list(dict)
                    """

        tweets = []
        error_count = 0
        for tid in tweet_ids:
            try:
                tweets.append(api.get_status(tid))
            except:
                error_count += 1
        print(f'Unable to fetch {error_count} tweets... using botometer lite on {len(tweets)-error_count} tweets')
        return self.request_botometer_lite_tweet_results_from_objects(tweets)

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

    def lite_results_to_dataframe(self, results, keep_tweet_ids=False):
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
        if not keep_tweet_ids:
            df = df.drop(labels=['tweet_id'], axis=1)
        return df

    def merge_result_and_original_dataframes(self, df_results, original, drop_nulls=False):
        pass
