import importlib
import tweepy
import pandas as pd

Utils = importlib.import_module('utilities').Utils

"""TweetCollector
 
"""


class TwitterManager:

    """Handles sentiment analysis for Twitter data, i.e. tweets. Can tag input data or a live stream of tweets.
    """

    def __init__(self):

        """Constructor method, creates a login session for the Twitter API, and sets up a stream listener.
        """

        # Tokens and keys
        #  TODO move keys to a hashed file on a cloud server
        self.shKey = 'BUYwpbmEi3A29cm9kOXeX9y8n'
        self.scKey = 'MF5w3g6jmn7WnYM6DG8xtIWkdjnEhInnBSf5bU6HclTF4wSkJ9'
        self.shToken = '4149804506-DrTR0UhuQ8pWf16r9wm8NYdkGNSBWuib2Y8nUlw'
        self.scToken = 'PYkfcY2w6tvovb4RMyCNpEAPyValmReJlaHUgC2KsWWzQ'
        self.bearer = 'AAAAAAAAAAAAAAAAAAAAAHfLGwEAAAAA23M6%2FAOdV1Gp6xMLfN1txD8DUwM%3DlxzfwHEkS1xPDAcAMncrZKVOUJEL0csMBxQCKOFp89CLlcVo6v'

        # Set up authentication
        self.auth = tweepy.OAuthHandler(self.shKey, self.scKey)
        self.auth.set_access_token(self.shToken, self.scToken)

        self.api = tweepy.API(self.auth)

    def phrase_search_history_with_id_jsons(self, phrase, count=1000):
        print('Collecting tweets from phrase: ' + phrase + '...')

        tweets = []
        tweet_keys = ['Tweet id', 'User id', 'Screen name', 'Label', 'Search term', 'json']
        while True:  # Iterate if tweet collection fails

            data = tweepy.Cursor(self.api.search,
                                 q=phrase,
                                 tweet_mode='extended',
                                 lang="en").items(count)

            print('Compiling tweets...')

            try:
                for tweet in data:

                    temp_dict = {}
                    temp_dict.update({tweet_keys[0]: tweet.id,
                                      tweet_keys[1]: tweet.user.id,
                                      tweet_keys[2]: tweet.user.screen_name,
                                      tweet_keys[3]: -2,
                                      tweet_keys[4]: phrase,
                                      tweet_keys[5]: tweet._json})

                    tweets.append(temp_dict)

                break
            except tweepy.error.TweepError as e:
                if 'code = 429' in e:
                    new_count = count - 100
                    print('Error 429 received, lowering history count from {} to {}.'.format(count, new_count))
                    count = new_count

        return pd.DataFrame(data=tweets, columns=tweet_keys)

    @staticmethod
    def construct_twitter_query(phrase, filter_in=None, filter_out=None, exact_phrase=''):

        """Constructs a proper advanced twitter search query given certain operations.
        Refer to https://developer.twitter.com/en/docs/twitter-api/v1/rules-and-filtering/overview/standard-operators

        :param phrase: The phrase to create the query with
        :type phrase: str
        :param filter_in: Types of data to include in query; defaults to []
        :type filter_in: list(str)
        :param filter_out: Types of data to remove from query; defaults to []
        :type filter_out: list(str)
        :param exact_phrase: An exact phrase to look for in the tweets; defaults to ''
        :type exact_phrase: str

        :return: The properly formatted query for a Twitter API call
        :rtype: str
        """

        # Default to [], to hide the mutable warning
        if not filter_in:
            filter_in = []
        if not filter_out:
            filter_out = []

        # Specify query phrasing
        query = ''
        if exact_phrase != '':
            query += '"' + exact_phrase + '" '

        query += phrase + ' OR ' + phrase.upper() + ' OR ' + phrase.capitalize()

        # Add all filtered in requirements
        for fin in filter_in:
            query += ' filter:' + fin

        # Add all filtered out requirements
        for fout in filter_out:
            query += ' -filter:' + fout

        return query


def prep_for_labeling(path, name):
    data = pd.read_csv(path)
    data = data.drop(columns='json')
    data['Label'] = -2


def collect_tweets(phrase='', filter_in=None, filter_out=None, history_count=1000):

    if filter_out is None:
        filter_out = ['retweets']
    if not filter_in:
        filter_in = []
    if not filter_out:
        filter_out = []

    tw = TwitterManager()

    # Search phrase
    query = tw.construct_twitter_query(phrase, filter_in=filter_in, filter_out=filter_out)
    tweets = tw.phrase_search_history_with_id_jsons(query, history_count)

    return tweets
