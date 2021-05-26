import botometer
import pandas as pd
import importlib
import os
from dotenv import load_dotenv
import time
Utils = importlib.import_module('utilities').Utils


OUTPUT_PATH = '../data/BotometerResults/'
load_dotenv('BotometerKeys.env')
rapidapi_key = os.getenv('rapidapi_key')
twitter_app_auth = {
    'consumer_key': os.getenv('consumer_key'),
    'consumer_secret': os.getenv('consumer_secret')
    }
ALL_BOT_KEYS = os.getenv('ALL_BOT_KEYS').split(',')

bom = botometer.Botometer(wait_on_ratelimit=True,
                          rapidapi_key=rapidapi_key,
                          **twitter_app_auth)


def request_botometer_results(user_ids):
    """Requests botometer scores and returns a list of dictionaries with the provided scores, cleaned for errors

            :param user_ids: list of Twitter user ids
            :type user_ids: list(str)

            :return: list of dictionaries of results from botometer, without error values
            :rtype: list(dict)
            """
    results = []
    for screen_name, result in bom.check_accounts_in(user_ids):
        results.append(result)

    return clean_errors(results)


def clean_errors(results):
    """Requests botometer scores and returns a list of dictionaries with the provided scores, cleaned for errors

                :param results: list of botometer result dictionaries
                :type user_ids: list(dict)

                :return: list of dictionaries of botometer results, without error values
                :rtype: list(dict)
                """
    clean_results = []
    for result in results:
        if 'error' in result.keys():
            pass
        else:
            clean_results.append(result)

    return clean_results


def get_unwanted_keys(wanted_keys):
    """Takes a list of wanted botometer keys and subtracts it from the list of all botometer keys to yield unwanted

                        :param wanted_keys: list of keys to store in dataframe
                        :type wanted_keys: list(str)

                        :return: list of unwanted keys
                        :rtype: list(str)
                        """
    return list(set(wanted_keys) ^ set(ALL_BOT_KEYS))


def results_to_dataframe(results, wanted_keys):
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
    dropped = all_keys.drop(labels=get_unwanted_keys(wanted_keys), axis=1)
    dropped.insert(0, "user_id", ids)
    return dropped


