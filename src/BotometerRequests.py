import botometer
import pandas
import pandas as pd
import os

import requests.exceptions
from dotenv import load_dotenv
import time
import itertools
import tweepy
import ast
import utilities
import TwitterManager


Utils = utilities.Utils
tm = TwitterManager.TwitterManager

# TODO Add wrapper functions for main entry points
# TODO Add functions for accuracy, recall, precision, F-scores
# TODO Add doc string comments for all methods, class, and the overall module
# @TODO reorganize columns in a df


class BotometerRequests:

    """
    Used to interact with Botometer and Botometer Lite. Initializes bom and lite connections.
    """
    def __init__(self, twitter_api=False):

        load_dotenv('../doc/BotometerKeys.env')
        
        self.rapidapi_key = os.getenv('rapidapi_key')
        self.twitter_app_auth = {
            'consumer_key': os.getenv('consumer_key'),
            'consumer_secret': os.getenv('consumer_secret')
            }
        self.ALL_BOT_KEYS = os.getenv('ALL_BOT_KEYS').split(',')
        self.ALL_LITE_KEYS = ['user_id', 'tweet_id', 'botscore']
        self.excel_keys = ['Tweet id', 'User id', 'user.majority_lang', 'user.user_data.screen_name', 'Label',
                           'botscore', 'u_botscore', 'cap.english', 'cap.universal', 'raw_scores.english.astroturf',
                           'raw_scores.english.fake_follower', 'raw_scores.english.financial',
                           'raw_scores.english.other', 'raw_scores.english.overall', 'raw_scores.english.self_declared',
                           'raw_scores.english.spammer', 'raw_scores.universal.astroturf',
                           'raw_scores.universal.fake_follower', 'raw_scores.universal.financial',
                           'raw_scores.universal.other', 'raw_scores.universal.overall',
                           'raw_scores.universal.self_declared', 'raw_scores.universal.spammer']

        self.bot_null_result = ast.literal_eval(os.getenv('null_bot_dictionary'))
        self.lite_null_result = {"botscore": -1, "tweet_id": -1, "user_id": -1}
        self.u_lite_null_result = {"u_botscore": -1, "tweet_id": -1, "user_id": -1}

        self.auth = None
        self.twitter_api = twitter_api
        self.file_keys = ['Tweet id', 'User id', 'json']

        self.bom = botometer.Botometer(wait_on_ratelimit=True,
                                       rapidapi_key=self.rapidapi_key,
                                       **self.twitter_app_auth)

        self.lite = botometer.BotometerLite(rapidapi_key=self.rapidapi_key, **self.twitter_app_auth)

    def setup_tweepy_api(self):
        """
        Initializes the tweepy API using keys from stored env file.
        """
        self.auth = tweepy.OAuthHandler(self.twitter_app_auth['consumer_key'], self.twitter_app_auth['consumer_secret'])
        self.auth.set_access_token(os.getenv('shToken'), os.getenv('scToken'))
        self.twitter_api = tweepy.API(self.auth, wait_on_rate_limit=True)

    @staticmethod
    def replace_english_with_universal(key: str) -> str:
        """
        Replaces the word "english" with "universal" in a string... May be used for generating Botometer universal keys
        given an english key.

        :param key: string key with english in it
        :type key: str

        :return: string with universal in it
        :rtype: str
        """
        a = key.find("english")
        if a == -1:
            return key
        b = a + len("english")
        return key[:a] + 'universal' + key[b:]

    def next_result_or_blank(self, result):
        """
        Returns either the Botometer score dictionary from a result or a blank version if there is an error.

        :param result: Result dictionary exactly as provided by bom.check_account
        :param type: dict

        :return: Score dictionary or a blank score dictionary (filled with -1)
        :rtype: dict
        """
        if 'error' == next(iter(result[1])):
            return self.bot_null_result
        else:
            return result[1]

    def request_botometer_results_map(self, user_ids):
        """
        Requests botometer scores and returns a map of dictionaries with the provided scores, errors replaced with -1.

        :param user_ids: list of Twitter user ids
        :type user_ids: list(str)

        :return: map of dictionaries of results from botometer, errors replaced with -1
        :rtype: map(dict)
        """
        while True:
            try:
                bom_generator = self.bom.check_accounts_in(user_ids)
                break
            except Exception as e:
                print(f'Some error {e} with botometer... Trying again in 1 second')
                time.sleep(1)
        return map(self.next_result_or_blank, bom_generator)

    def request_botometer_results(self, user_ids):
        """
        Requests botometer scores and returns a list of dictionaries with the provided scores, cleaned for errors.

        :param user_ids: list of Twitter user ids
        :type user_ids: list(str)

        :return: list of dictionaries of results from botometer, without null dicts instead of error values
        :rtype: list(dict)
        """

        tries = 0
        error_count = 0
        results = []

        while tries < 20:
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
            except requests.exceptions.HTTPError as e:
                if '429 Client Error' in str(e):
                    limit_unscored = len(user_ids) - len(results)
                    print(f'Botometer requests daily limit met, unable to score {limit_unscored} users')

                    # Pad results with null entries for unscored users
                    results += ([self.bot_null_result] * limit_unscored)
                    error_count += limit_unscored
                    break

                else:
                    print(f'Some HTTP Error {e} with botometer... Trying again in 2 seconds')
                    tries += 1
                    time.sleep(2)

            except Exception as e:
                print(f'Some error {e} with botometer... Trying again in 2 seconds')
                tries += 1
                time.sleep(2)

        if len(results) < len(user_ids):
            unscored = len(user_ids) - len(results)
            # Pad results with null entries for unscored users
            results += ([self.bot_null_result] * unscored)
            error_count += unscored

        print(f'Botometer User Request Complete: '
              f'Unable to score {error_count} users... {len(user_ids)-error_count} users scored')
        return results

    @staticmethod
    def clean_errors(results, err_str='Error scores removed: '):
        """
        Removes error instances from a list of botometer results.

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
        """
        Replaces error instances from a list of botometer results with -1 dictionary skeleton.

        :param results: list of botometer result dictionaries
        :type results: list(dict)

        :return: list of dicts of botometer results, with error values as -1 filled dicts
        :rtype: list(dict)
        """
        return [res if 'error' != next(iter(res)) else null_result for res in results]

    def get_unwanted_keys(self, wanted_keys):
        """
        Takes a list of wanted botometer keys and subtracts it from the list of all botometer keys to yield unwanted.

        :param wanted_keys: list of keys to store in dataframe
        :type wanted_keys: list(str)

        :return: list of unwanted keys
        :rtype: list(str)
        """
        return list(set(wanted_keys) ^ set(self.ALL_BOT_KEYS))

    def results_to_dataframe(self, results, wanted_keys):
        """
        Takes a list of botometer results and converts them to a pandas dataframe placing user ids at the front
        followed by wanted keys.

        :param results: list of botometer result dictionaries
        :type user_ids: list(dict)

        :param wanted_keys: list of keys to store in dataframe
        :type wanted_keys: list(str)

        :return: pandas dataframe with column id and wanted keys columns
        :rtype: pandas dataframe size: [1 + wanted_keys x len(results)]
        """

        all_keys = pd.json_normalize(results)
        ids = all_keys['user.user_data.id_str']
        dropped = all_keys.drop(labels=self.get_unwanted_keys(wanted_keys), axis=1, errors='ignore')
        dropped.insert(0, "user_id", ids)
        
        return dropped

    @staticmethod
    def min_of_columns(df_results: pandas.DataFrame, columns):
        """
        @TODO move to utils file
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
    def max_of_columns(df_results: pandas.DataFrame, columns):
        """
        @TODO move to utils file
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
    def classify_results(data, threshold):
        """
        Takes a list of values and returns a label list based on the threshold.

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
        """
        Takes a list of thresholds and a list of classifiers and returns a comprehensive list of each classifier with
        each threshold:
        [botscore, u_botscore] & [0.5, 0.6] --> [botscore.0.5, u_botscore.0.5, botscore.0.6, u_botscore.0.6].

        :param thresholds: a list of thresholds to be applied
        :type thresholds: list(float)
        :param classifiers: a list of classifiers in the form of key strings from the applicable dataframe
        :type classifiers: list(str)

        :return: list of threshold values, classifiers, and names with matching indices
        :rtype: list(float), list(str), list(str)
        """

        new_thresholds = [t for t in thresholds for c in classifiers]
        new_classifiers = [c for t in thresholds for c in classifiers]
        names = [new_classifiers[i] + '.' + str(new_thresholds[i]) for i in range(len(new_classifiers))]
        return new_thresholds, new_classifiers, names

    def classify_one_to_one(self, df_results, thresholds, classifiers, names):
        """
        Takes a dataframe of scores, a list of thresholds, a list of classifier keys, and a list of names for the label
        columns and classifies them one to one (each classifier with each threshold according to index). Returns a
        dataframe of labels.

        :param df_results: a dataframe with classifiers and float scores
        :type df_results: pandas.DataFrame
        :param thresholds: a list of float spam thresholds
        :type thresholds: list(float)
        :param classifiers: a list of classifier keys (column names from df_results)
        :type classifiers: list(str)
        :param names: a list of new column names for the resulting label columns
        :type names: list(str)

        :return: A dataframe full of labels for the threholds and classifiers
        :rtype: pandas.DataFrame()
        """
        labels = {n: self.classify_results(df_results[c], t) for c, t, n in zip(classifiers, thresholds, names)}
        return pd.DataFrame(labels)

    def classify_for_each_threshold(self, df_results, thresholds, classifiers):
        """
        Takes a dataframe of scores, a list of thresholds, and a list of classifier keys and returns a dataframe of
        labels with classification for each threshold-classifier combination.

        :param df_results: a dataframe with classifiers and float scores
        :type df_results: pandas.DataFrame
        :param thresholds: a list of float spam thresholds
        :type thresholds: list(float)
        :param classifiers: a list of classifier keys (column names from df_results)
        :type classifiers: list(str)

        :return: A dataframe full of labels for each threshold-classifier pair
        :rtype: pandas.DataFrame()

        """
        new_thresholds, new_classifiers, names = self.thresholds_x_classifiers(thresholds, classifiers)
        return self.classify_one_to_one(df_results, new_thresholds, new_classifiers, names)

    @staticmethod
    def segment_list(lst, n):
        """
        @TODO move to utils file
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
        @TODO move to utils file
        Flattens a list of lists into one list
        :param lst: list of lists to flatten
        :type lst: list

        :return: flattened list
        :rtype: list
        """
        return list(itertools.chain.from_iterable(lst))

    def request_botometer_lite_tweet_results_from_objects(self, tweets):
        """
        Takes a list of json tweet objects (either as dictionaries or strings) and returns a list of Botometer lite
        results.

        :param tweets: list of json tweet objects
        :type tweets: list(dict) or list(str)

        :return: list of Botometer lite results
        :rtype: list(dict)
        """

        if type(tweets[0]) == str:
            tweets = self.strings_to_dicts(tweets)
        if len(tweets) > 100:
            scores = self.flatten([self.lite.check_accounts_from_tweets(x) for x in self.segment_list(tweets, 100)])
        else:
            scores = self.lite.check_accounts_from_tweets(tweets)

        return self.replace_errors(scores, self.lite_null_result)

    @staticmethod
    def get_tweets_from_ids(tweet_ids, api):
        """
        Takes a list of tweet ids and Tweepy api and returns a list of json tweet objects.

        :param tweet_ids: list of tweet ids
        :type tweet_ids: list(str)

        :param api: Initialized tweepy API, use self.setup_tweepy_api and self.twitter_api or external api
        :type api: tweepy.API

        :return: list of json tweet objects
        :rtype: list(dict)
        """

        tweets = []
        error_count = 0
        for tid in tweet_ids:
            try:
                tweets.append(api.get_status(tid)._json)
                print(f'Tweet number: {len(tweets) + error_count}')
            except Exception as e:
                tweets.append(-1)
                print(f'Tweet number: {len(tweets) + error_count} error {e}.')
                error_count += 1
        print(f'Unable to fetch {error_count} tweets... fetched {len(tweets)} tweets')

        return tweets

    def request_botometer_lite_user_results(self, user_ids):
        """
        Takes a list of user ids and returns a list of Botometer lite results for those users.

        :param user_ids: list of user ids
        :type user_ids: list(str)

        :return: list of Botometer lite results
        :rtype: list(dict)
        """

        if len(user_ids) > 100:
            scores = self.flatten([self.lite.check_accounts_from_user_ids(x) for x in self.segment_list(user_ids, 100)])

        else:
            scores = self.lite.check_accounts_from_user_ids(user_ids)

        return self.replace_errors(scores, self.lite_null_result)

    def lite_results_to_dataframe(self, results, keep_tweet_ids=True):
        """
        Takes a list of Botometer lite results and returns a pandas dataframe.

        :param results: list of Botometer lite results
        :type results: list(dict)

        :param keep_tweet_ids: whether or not to keep the tweet_ids
        :type keep_tweet_ids: bool

        :return: pandas dataframe with user_id, tweet_id, botscore columns
        :rtype: pandas.DataFrame
        """

        df = pd.json_normalize(results)
        df = df[self.ALL_LITE_KEYS]

        if not df['tweet_id'][0]:
            df = df.rename(columns={'botscore': 'u_botscore'})

        if not keep_tweet_ids:
            df = df.drop(labels=['tweet_id'], axis=1)
        return df

    @staticmethod
    def strings_to_dicts(strings):
        """
        @TODO move to utils file
        Converts a list of json style strings toa list of dictionaries.

        :param strings: list of strings
        :type strings: list(str)

        :return: list of dicts
        :rtype: list(dict)

        """
        return [ast.literal_eval(s) for s in strings]

    def merge_dataframes(self, original, new, original_id_key, new_id_key, new_data_keys):
        """
        @TODO move to utils file
        Merges two dataframes with replacement of missing values in the new according to an id column in both dataframes.

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
        @TODO add to utils file
        Merges two pandas dataframe of equal size

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
        @TODO add to utils file
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
    def replace_missed(expected, result_indices, results, null_result=-1):
        """
        @TODO review for deletion
        Inserts some null value into a list of data at spots where result_indices do not line up with expected.

        :param expected: list of expected ids
        :type expected: list

        :param result_indices: list of resulting ids
        :type result_indices: list

        :param results: list of results with same indices as result_indices
        :type results: list

        :param null_result: null value to insert instead of missing data
        :type null_result: any

        :return: data with nulls inserted for missing values
        :rtype: list
        """
        if len(expected) == len(result_indices):
            return results
        for i in range(len(expected)):
            if result_indices[i] != expected[i]:
                result_indices.insert(i, expected[i])
                results.insert(i, null_result)

        return results

    @staticmethod
    def calculate_measures(results, labels):
        """
        Calculates the accuracy, precision, recall, and f1 score from a list of result labels and actual labels.

        :param results: list of result labels
        :type results: list(int)

        :param labels: list of actual labels
        :type labels: list(int)

        :return: accuracy, precision, recall, f1 score
        :rtype: float, float, float, float
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

        accuracy = (tp+tn)/(tp+fp+fn+tn-sk)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = 2 * (recall * precision) / (recall + precision)

        return accuracy, precision, recall, f1, (tp, fp, tn, fn)

    @staticmethod
    def calc_averages(results, labels):
        """
        Calculates the average clean and spam value from results according to labels.

        :param results: list of float values representing scores
        :type results: list(float)

        :param labels: list of actual labels
        :type labels: list(int)

        :return: clean average and spam average result value
        :rtype: float, float
        """
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

    @staticmethod
    def order_dataframe(dataframe, new_order):
        """
        Changes the order of a dataframe's columns using the new order

        :param dataframe: dataframe
        :type dataframe: pd.DataFrame

        :param new_order: The new order of columns as a list of strings.
        :type new_order: list(str)

        :return: reordered dataframe
        :rtype: pd.DataFrame
        """
        dataframe = dataframe[new_order]
        return dataframe

    def to_excel_analysis_format(self, dataframe):
        dataframe = self.order_dataframe(dataframe, self.excel_keys)
        dataframe['Tweet id'] = ["'" + str(id) for id in dataframe['Tweet id'].tolist()]
        dataframe['User id'] = ["'" + str(id) for id in dataframe['User id'].tolist()]
        return dataframe

    def wrapper(self, from_dataframe=False, from_file=False, to_file=False, tweet_ids=None, tweet_objects=None,
                user_ids=None, bot_user_request=False, lite_user_request=False, lite_tweet_request=False,
                thresholds=None, classifiers=None, existing_api=None, wanted_bot_keys=None):

        """
        A wrapper method to work with BotometerRequests.

        :param from_dataframe: a dataframe with data to label
        :type: pd.DataFrame

        :param from_file: a string filepath to a csv file to use as the input data. Data column keys should match
                            self.file_keys
        :type from_file: str

        :param to_file: a str filepath to a csv file to export the results
        :type to_file: str

        :param tweet_ids: a list of tweet ids to request lite scores for (slows down speed significantly due to
                            Twitter rate limits)
        :type tweet_ids: list(str) or list(int)

        :param tweet_objects: a list of tweet json objects to request lite scores for
        :type tweet_objects: list(dict) or list(str)

        :param user_ids: a list of user ids to request lite user or bom scores for
        :type user_ids: list(str) or list(int)

        :param bot_user_request: whether or not to make a Botometer user request (slows down speed significantly due to
                                Twitter rate limits)
        :type bot_user_request: bool

        :param lite_user_request: whether or not to make a Botometer lite user request based on user ids
        :type lite_user_request: bool

        :param lite_tweet_request: whether or not to make a Botometer lite tweet request based on tweet ids or objects
        :type lite_tweet_request: bool

        :param thresholds: list of thresholds to use for labeling
        :type thresholds: list(float)

        :param classifiers: list of classifier keys to use for labeling
        :type classifiers: list(str)

        :param existing_api: an existing Tweepy.API instance to pass to Botometer or use for collecting tweet jsons
        :type existing_api: Tweepy.API

        :param wanted_bot_keys: a list of wanted keys (scores to keep) for Botometer user request
        :type wanted_bot_keys: list(str)
        """

        #  Pulls data from input file to use for Botometer data requests
        existing_file = False
        if from_file or from_dataframe is not False:
            existing_file = True
            if from_file:
                from_file = pd.read_csv(from_file)
            else:
                from_file = from_dataframe

            cols = [[], [], []]
            for ci in range(len(cols)):
                try:
                    cols[ci] = from_file[self.file_keys[ci]].tolist()
                except KeyError:
                    cols[ci] = None

            tweet_ids, user_ids, tweet_objects = cols

        #  In case you don't have the data, catches error request to Botometer
        if user_ids is None:
            bot_user_request = False
            lite_user_request = False

        #  Grabs tweet objects if available, otherwise looks up objects using Twitter API
        if tweet_objects is None:
            
            #  In case you don't have the data, catches error request to Botometer lite
            if tweet_ids is None:
                lite_tweet_request = False
            
            #  Otherwise, grabs Tweet objects from the API
            elif lite_tweet_request:
                if existing_api is None:
                    self.setup_tweepy_api()
                    api = self.twitter_api
                else:
                    api = existing_api
                    
                print("Collecting tweet jsons from ids... Might take a while")
                tweet_objects = self.get_tweets_from_ids(tweet_ids, api)
                
                #  If existing file exists, will add the JSON objects
                if existing_file:
                    from_file['json'] = tweet_objects
                    
        #  If no Tweet IDs, extracts the IDs to be used as a column from the tweet objects
        elif tweet_ids is None:
            tweet_ids = [obj['id'] for obj in tweet_objects]
            if existing_file:
                from_file['Tweet id'] = tweet_ids

        #  If no wanted keys given, set to grab english
        if wanted_bot_keys is None:
            wanted_bot_keys = ['cap.english']

        #  If no thresholds or classifiers given, don't classify
        if thresholds is None or classifiers is None:
            classify = False
        else:
            classify = True

        result_df = pd.DataFrame()

        #  If a lite request, then use Botometer lite
        if lite_tweet_request:
            print('Handling Lite Tweet Request...')
            lite_results = self.request_botometer_lite_tweet_results_from_objects(tweet_objects)
            temp_df = self.lite_results_to_dataframe(lite_results)
            result_df['Tweet id'] = tweet_ids
            result_df = self.merge_dataframes(result_df, temp_df, 'Tweet id', 'tweet_id', ['botscore'])

        #  If a lite user request, then use Botometer lite on users
        if lite_user_request:
            print('Handling Lite User Request...')
            lite_user_results = self.request_botometer_lite_user_results(user_ids)
            temp_df = self.lite_results_to_dataframe(lite_user_results)
            result_df['User id'] = user_ids
            result_df = self.merge_dataframes(result_df, temp_df, 'User id', 'user_id', ['u_botscore'])

        #  If a regular user request, then use Botometer
        if bot_user_request:
            print('Handling Botometer User Request...')
            bot_user_results = self.request_botometer_results(user_ids)
            temp_df = self.results_to_dataframe(bot_user_results, wanted_bot_keys)
            temp_df = temp_df.drop('user_id', axis=1)
            result_df = self.basic_merge(result_df, temp_df)

        # If specified to classify, classify for each threshold
        if classify:
            label_df = self.classify_for_each_threshold(result_df, thresholds, classifiers)
            result_df = self.basic_merge(result_df, label_df)

        #  Write to output file
        if to_file:
            if existing_file:
                write_df = self.basic_merge(from_file, result_df)
            else:
                write_df = result_df

            write_df.to_csv(to_file, index=False)

        return result_df
