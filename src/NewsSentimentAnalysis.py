import importlib
import tweepy
from bs4 import BeautifulSoup
import requests
import pandas as pd
import nltk
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords

Utils = importlib.import_module('utilities').Utils
NSC = importlib.import_module('NLPSentimentCalculations').NLPSentimentCalculations


"""NewsSentimentAnalysis

Description:
Module for working with News Sentiment Analysis. Uses the NLPSentimentCalculations module for all the backend generic
calls, like data sanitization. This works with data gathering, filtering, and I/O. Current focus is on news headlines,
tweets, and articles. Can support data streams and data input classification.

Authors: Benjamin Collins
Date: April 22, 2021 
"""


class NewsManager:

    """Handles news headline and article classification.
    """

    @staticmethod
    def get_ticker_news_headlines(ticker):

        """Given a ticker, scrapes finviz for news headlines.

        :param ticker: Ticker to scan for news headlines about
        :type ticker: str

        :return: A list of news headlines regarding the ticker
        :rtype: list(str)
        """

        url = 'https://finviz.com/quote.ashx?t=' + ticker

        r = requests.get(url, headers={'user-agent': 'my-app/0.0.1'})
        html = BeautifulSoup(r.text, 'html.parser')

        # Parse HTML for headlines
        data = []
        news_rows = html.find(id='news-table').find_all('tr')
        for i, row in enumerate(news_rows):

            # Get date and time posted
            time_data = row.td.text.split()

            if len(time_data) == 1:
                time = time_data[0]
                date = 'NA'
            else:
                date = time_data[0]
                time = time_data[1]

            data.append((date, time, row.a.text))

        return data

    def get_tickers_news_headlines(self, tickers):

        """Gets news headlines for a list of tickers

        :param tickers: List of tickers to scan for news headlines about
        :type tickers: list(str)

        :return: A dictionary of tickers to dataframes of news headlines
        :rtype: dict(str-> :class:`pandas.core.frame.DataFrame`)
        """

        data = {}
        for ticker in tickers:
            data[ticker] = pd.DataFrame(self.get_ticker_news_headlines(ticker))

        return data


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

        # Set up stream listener
        self.listener = TwitterStreamListener()
        self.stream = tweepy.Stream(auth=self.auth, listener=self.listener)

        self.nsc = NSC()

    def initialize_nltk_twitter(self):

        """Uses a basic NLTK dataset to train and test a positive/negative binary classifier.
        """

        # Download all the common NLTK data samples
        nltk.download('twitter_samples')
        nltk.download('stopwords')

        p_tweets_tokens = twitter_samples.tokenized('positive_tweets.json')
        p_dataset = TwitterManager.get_dataset_from_tweet(p_tweets_tokens, 'Positive')

        n_tweets_tokens = twitter_samples.tokenized('negative_tweets.json')
        n_dataset = TwitterManager.get_dataset_from_tweet(n_tweets_tokens, 'Negative')

        dataset = p_dataset + n_dataset
        Utils.shuffle_list(dataset)

        split_count = int(len(dataset) * 0.7)  # Does a 70:30 split for train/test data
        train_data = dataset[:split_count]
        test_data = dataset[split_count:]

        self.nsc.train_naivebayes_classifier(train_data)
        self.nsc.test_classifier(test_data)

    @staticmethod
    def get_dataset_from_tweet(tweet_tokens, classifier_tag):

        """Given a list of tweet tokens, clean the tweets into tokens with removed stopwords, bad ascii, and so on. For
        the full list, see NLPSentimentCalculations::get_clean_tokens. Then, splits them into a list of dictionaries
        for easy token manipulation and training on a classifier.

        :param tweet_tokens: List of tweet tokens, which is itself a list of strings
        :type tweet_tokens: list(list(str))
        :param classifier_tag: A classification label to mark all tokens
        :type classifier_tag: str

        :return: A list of tuples of dictionaries of feature mappings to classifiers
        :rtype: list(dict(str-> bool), str)
        """

        tweets_clean = NSC.get_clean_tokens(tweet_tokens, stopwords.words('english'))
        return NSC.get_basic_dataset(tweets_clean, classifier_tag)

    def get_tweet_sentiment(self, text):

        """Calls the model classifier on a tweet to get the sentiment.

        :param text: List of tweet tokens, which is itself a list of strings
        :type text: str

        :return: A list of tuples of dictionaries of feature mappings to classifiers
        :rtype: list(dict(str-> bool), str)
        """

        sentiment = self.nsc.classify_text(text)
        print('Text:', text)
        print('Sentiment:', sentiment)
        return sentiment

    def phrase_search_history(self, phrase, count=1000):

        """Searches historic tweets for a phrase

        :param phrase: A phrase to search for
        :type phrase: str
        :param count: Amount of tweets to grab; defaults to 1000
        :type count: int

        :return: A dataframe of tweets and data that match the phrase
        :rtype: :class:`pandas.core.frame.DataFrame`
        """

        print('Collecting tweets from phrase: ' + phrase + '...')

        tweets = []
        tweet_keys = ['Date', 'User', 'Text', 'Sentiment']

        while True:  # Iterate if tweet collection fails

            data = tweepy.Cursor(self.api.search,
                                 q=phrase,
                                 tweet_mode='extended',
                                 lang="en").items(count)

            print('Compiling tweets...')

            try:
                for tweet in data:

                    if self.nsc.classifier is not None:
                        sentiment = self.get_tweet_sentiment(tweet.full_text)
                    else:
                        sentiment = 'None'

                    temp_dict = {}
                    temp_dict.update({tweet_keys[0]: tweet.created_at, tweet_keys[1]: tweet.user.name,
                                      tweet_keys[2]: tweet.full_text, tweet_keys[3]: sentiment})

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

    def start_stream(self, phrases, limit=-1):

        """Starts a stream to collect tweets based on search phrases

        :param phrases: The phrases to search for in a stream
        :type phrases: list(str)
        :param limit: The number of tweets to limit in your stream scanning, currently unused; defaults to -1
        :type limit: int

        :return: The properly formatted query for a Twitter API call
        :rtype: str
        """

        print('Starting stream on ' + str(phrases))

        # Set filename
        filename = ''
        for phrase in phrases:
            filename += phrase + "_"
        self.listener.output_file = '../data/' + filename + 'tweets_stream.csv'

        self.stream.filter(track=phrases, is_async=True)


class TwitterStreamListener(tweepy.StreamListener):

    """Handles the stream event-driven methods, inherits tweepy.StreamListener
    """

    def __init__(self):

        """Constructor method, calls parent class initializer
        """

        super(TwitterStreamListener, self).__init__()
        self.header_written = False
        self.output_file = ''

    def on_status(self, status):

        """Main event method for when a stream listener gets a tweet. Writes to a file.

        :param status: The event status data from the listener stream
        :type status: :class:`tweepy.api.API`
        """

        # Set write type, where 'w' is write from scratch and 'a' is append
        if not self.header_written:
            write_type = 'w'
        else:
            write_type = 'a'

        with open(self.output_file, write_type, encoding='utf-8') as f:

            # If no header has been written, write
            if not self.header_written:
                f.write('Date,User,Text,\n')
                self.header_written = True

            data = str(status.created_at) + ',' + status.user.name + ',' + status.text.replace('\n', '') + ',\n'
            print(data)
            f.write(data)
            print('Tweet saved...')


def main(phrase='', filter_in=None, filter_out=None, history_count=1000):

    if not filter_in:
        filter_in = []
    if not filter_out:
        filter_out = []

    tw = TwitterManager()

    tw.initialize_nltk_twitter()
    #tw.get_tweet_sentiment('Tesla can suck it')

    # Search phrase
    query = tw.construct_twitter_query(phrase, filter_in=filter_in, filter_out=filter_out)
    tweets = tw.phrase_search_history(query, history_count)
    Utils.write_dataframe_to_csv(tweets, '../data/News Sentiment Analysis/'
                                         '' + phrase + '_tweet_history_search.csv')

    # Start query stream
    #tw.start_stream(['$GME'])


if __name__ == '__main__':

    main()
