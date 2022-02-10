import tweepy
from bs4 import BeautifulSoup
import requests
import pandas as pd
import nltk
from nltk.corpus import twitter_samples
import utilities
import NLPSentimentCalculations
import TwitterSpamModel
import TwitterSentimentModel
from SpamToSentimentModel import ModelHandler

Utils = utilities.Utils
NSC = NLPSentimentCalculations.NLPSentimentCalculations
SpamModelParameters = TwitterSpamModel.SpamModelParameters
SpamModelData = TwitterSpamModel.SpamModelData
SpamModelLearning = TwitterSpamModel.SpamModelLearning
SentimentModelParameters = TwitterSentimentModel.SentimentModelParameters
SentimentModelData = TwitterSentimentModel.SentimentModelData
SentimentModelLearning = TwitterSentimentModel.SentimentModelLearning


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

    def get_dataset_from_tweet_spam(self, dataframe, features_to_train=None):

        """Converts the text feature to a dataset of labeled unigrams and bigrams.

        :param dataframe: A dataframe containing the text key with all the text features to parse
        :type dataframe: :class:`pandas.core.frame.DataFrame`
        :param features_to_train: The list of all features to train on, does not need to include 'Label'
        :type features_to_train: list(str)

        :return: A list of tuples of dictionaries of feature mappings to classifiers
        :rtype: list(dict(str-> bool), str)
        """

        if not features_to_train:
            features_to_train = ['full_text']

        x_train, x_test, y_train, y_test = NSC.keras_preprocessing(dataframe[features_to_train], dataframe['Label'],
                                                                   test_size=0.185,
                                                                   augmented_states=dataframe['augmented'])

        if x_train is False:
            print("Train test failed due to over augmentation")
            return False

        # Split into text and meta data
        if 'full_text' in features_to_train:
            x_train_text_data = x_train['full_text']
            x_test_text_data = x_test['full_text']

            features_to_train.remove('full_text')
        else:
            x_train_text_data = pd.DataFrame()
            x_test_text_data = pd.DataFrame()

        # Clean the textual data
        x_train_text_clean = [NSC.sanitize_text_string(s) for s in list(x_train_text_data)]
        x_test_text_clean = [NSC.sanitize_text_string(s) for s in list(x_test_text_data)]

        # Initialize tokenizer on training data
        self.nsc.tokenizer.fit_on_texts(x_train_text_clean)

        # Create word vectors from tokens
        x_train_text_embeddings = self.nsc.keras_word_embeddings(x_train_text_clean)
        x_test_text_embeddings = self.nsc.keras_word_embeddings(x_test_text_clean)

        glove_embedding_matrix = self.nsc.create_glove_word_vectors()

        x_train_meta = x_train[features_to_train]
        x_test_meta = x_test[features_to_train]

        return x_train_text_embeddings, x_test_text_embeddings, x_train_meta, x_test_meta, \
               glove_embedding_matrix, y_train, y_test

    def initialize_twitter_spam_model(self, spam_model_learning: SpamModelLearning, to_preprocess_binary='', from_preprocess_binary='',
                                      learning_data='../data/Learning Data/spam_learning.csv', epochs=200,
                                      aug_df_file='',
                                      early_stopping=False, load_model=False,
                                      model_checkpoint_path='../data/analysis/Model Results/Saved Models/'
                                                            'best_spam_model.h5'):

        """Initializes, trains, and tests a Twitter spam detection model.
        """

    def initialize_twitter_sentiment_model(self):

        """Uses a basic NLTK dataset to train and test a positive/negative binary classifier.
        """

        # TODO use transfer learning with spam model

        # Download all the common NLTK data samples
        nltk.download('twitter_samples')
        nltk.download('stopwords')

        p_tweets_tokens = twitter_samples.tokenized('positive_tweets.json')
        p_dataset = TwitterManager.get_dataset_from_tweet(p_tweets_tokens, 'Positive')

        n_tweets_tokens = twitter_samples.tokenized('negative_tweets.json')
        n_dataset = TwitterManager.get_dataset_from_tweet(n_tweets_tokens, 'Negative')

        dataset = p_dataset + n_dataset
        Utils.shuffle_data(dataset)

        split_count = int(len(dataset) * 0.7)  # Does a 70:30 split for train/test data

        # TODO add spam dataset but only to train_data! Only add non-spam, as spam should be filtered out at this stage
        train_data = dataset[:split_count]
        test_data = dataset[split_count:]

        # TODO update to use Tensorflow
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

        tweets_clean = NSC.get_clean_tokens(tweet_tokens)
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

    def start_stream(self, phrases, default_sentiment='', limit=-1):

        """Starts a stream to collect tweets based on search phrases

        :param phrases: The phrases to search for in a stream
        :type phrases: list(str)
        :param default_sentiment: The default sentiment to assign to all incoming tweets; defaults to empty
        :type default_sentiment: str
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
        self.listener.default_sentiment = default_sentiment

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
        self.default_sentiment = ''

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

            if self.default_sentiment != '':
                self.default_sentiment += ','

            data = str(status.created_at) + ',' + status.user.name + ',' + status.text.replace('\n', '') +\
                   ',' + self.default_sentiment + '\n'

            print(data)
            f.write(data)
            print('Tweet saved...')


def main(search_past=False, search_stream=False, train_spam=False, train_sent=False,
         phrase='', filter_in=None, filter_out=None, history_count=1000):
    if not filter_in:
        filter_in = []
    if not filter_out:
        filter_out = []

    tw = TwitterManager()

    if train_spam:
        spam_model_params = SpamModelParameters(epochs=10000,
                                                batch_size=128,
                                                load_model=False,
                                                checkpoint_model=True,
                                                saved_model_bin='../data/analysis/Model Results/Saved Models/'
                                                                'best_spam_model.h5')

        spam_model_data = SpamModelData(nsc=NSC(), base_data_csv='../data/Learning Data/spam_learning.csv',
                                        test_size=0.01,
                                        features_to_train=['full_text', 'cap.english', 'cap.universal',
                                                           'raw_scores.english.overall',
                                                           'raw_scores.universal.overall',
                                                           'raw_scores.english.astroturf',
                                                           'raw_scores.english.fake_follower',
                                                           'raw_scores.english.financial',
                                                           'raw_scores.english.other',
                                                           'raw_scores.english.self_declared',
                                                           'raw_scores.english.spammer',
                                                           'raw_scores.universal.astroturf',
                                                           'raw_scores.universal.fake_follower',
                                                           'raw_scores.universal.financial',
                                                           'raw_scores.universal.other',
                                                           'raw_scores.universal.self_declared',
                                                           'raw_scores.universal.spammer',
                                                           'botscore', 'favorite_count', 'retweet_count'])

        spam_model_learning = SpamModelLearning(spam_model_params, spam_model_data)
        spam_model_learning.build_model()

        spam_score_dict = spam_model_learning.predict_and_score('../data/Learning Data/spam_test_set.csv')

        print(spam_score_dict)

    if train_sent:
        sentiment_model_params = SentimentModelParameters(epochs=150,
                                                          batch_size=128,
                                                          load_model=False,
                                                          checkpoint_model=True,
                                                          saved_model_bin='../data/analysis/Model Results/Saved Models/'
                                                                          'best_sentiment_model.h5')

        sentiment_model_data = SentimentModelData(nsc=NSC(), base_data_csv='../data/Learning Data/'
                                                                           'sentiment_learning.csv',
                                                  test_size=0.1)

        sentiment_model_learning = SentimentModelLearning(sentiment_model_params, sentiment_model_data)
        sentiment_model_learning.build_model()

        MH = ModelHandler(spam_model=spam_model_learning, sentiment_model=sentiment_model_learning)
        MH.analyze_tweets('SOURCE OF TWEETS', out_path='SOURCE TO WRITE TO')

    # Search phrase
    if search_past:

        query = tw.construct_twitter_query(phrase, filter_in=filter_in, filter_out=filter_out)
        tweets = tw.phrase_search_history(query, history_count)

        # Writes the file to csv and creates appropriate directories (if non-existent).
        # If failed, writes data to current directory to avoid data loss
        if not Utils.write_dataframe_to_csv(tweets, '../data/News Sentiment Analysis/'
                                                    '' + phrase + '_tweet_history_search.csv'):
            Utils.write_dataframe_to_csv(tweets, '' + phrase + '_tweet_history_search.csv')

    if search_stream:

        # Start query stream
        if isinstance(phrase, list):
            tw.start_stream(phrase)
        else:
            tw.start_stream(([phrase]))


main(train_spam=True)
