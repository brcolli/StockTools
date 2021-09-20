import importlib
import tweepy
from bs4 import BeautifulSoup
import requests
import pandas as pd
import nltk
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
import pickle
import os

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

        # if 'augmented' not in dataframe.keys():
        #     dataframe['augmented'] = 0
        #
        # if augmented_df is not None:
        #     if 'augmented' not in augmented_df.keys():
        #         augmented_df['augmented'] = 1
        #     dataframe = pd.concat([dataframe, augmented_df])

        x_train, x_test, y_train, y_test = NSC.keras_preprocessing(dataframe[features_to_train], dataframe['Label'],
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

    def initialize_twitter_spam_model(self, to_preprocess_binary='', from_preprocess_binary='',
                                      learning_data='../data/Learning Data/spam_learning.csv', epochs=100,
                                      aug_df_file='',
                                      early_stopping=False, load_model=False,
                                      model_checkpoint_path='../data/analysis/Model Results/Saved Models/'
                                                            'best_spam_model.h5'):

        """Initializes, trains, and tests a Twitter spam detection model.
        """

        if os.path.exists(from_preprocess_binary):

            with open(from_preprocess_binary, "rb") as fpb:
                data = pickle.load(fpb)
            
            x_train_text_embeddings, x_test_text_embeddings, x_train_meta, x_test_meta, \
                glove_embedding_matrix, y_train, y_test, self.nsc.tokenizer = data

        else:

            features_to_train = ['full_text']
            json_features = ['full_text']

            twitter_df = Utils.parse_json_botometer_data(learning_data, json_features)

            if 'augmented' not in twitter_df.columns:
                twitter_df['augmented'] = 0

            if aug_df_file:
                aug_df = Utils.parse_json_botometer_data(aug_df_file, json_features)
                if 'augmented' not in aug_df.columns:
                    aug_df['augmented'] = 1

                twitter_df = pd.concat([twitter_df, aug_df])

            x_train_text_embeddings, x_test_text_embeddings, x_train_meta, x_test_meta, \
            glove_embedding_matrix, y_train, y_test = self.get_dataset_from_tweet_spam(twitter_df, features_to_train)

            if to_preprocess_binary:
                data = (x_train_text_embeddings, x_test_text_embeddings, x_train_meta, x_test_meta,
                        glove_embedding_matrix, y_train, y_test, self.nsc.tokenizer)
                with open(to_preprocess_binary, "wb") as tpb:
                    pickle.dump(data, tpb)

        # Load previously saved model and test
        if load_model and os.path.exists(model_checkpoint_path):

            spam_model = NSC.load_saved_model(model_checkpoint_path)

            score = spam_model.evaluate(x=[x_test_text_embeddings, x_test_meta], y=y_test, verbose=1)

            print("Test Score:", score[0])
            print("Test Accuracy:", score[1])

            return spam_model

        spam_model = self.nsc.create_text_meta_model(glove_embedding_matrix,
                                                     len(x_train_meta.columns), len(x_train_text_embeddings[0]))

        # Print model summary
        spam_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        print(spam_model.summary())

        cbs = []
        if early_stopping:

            # Set up early stopping callback
            cbs.append(NSC.create_early_stopping_callback('accuracy', patience=10))

            cbs.append(NSC.create_model_checkpoint_callback(model_checkpoint_path, monitor_stat='accuracy'))

        if len(x_train_meta.columns) < 1:
            train_input_layer = [x_train_text_embeddings]
            test_input_layer = [x_test_text_embeddings]
        else:
            train_input_layer = [x_train_text_embeddings, x_train_meta]
            test_input_layer = [x_test_text_embeddings, x_test_meta]

        history = spam_model.fit(x=train_input_layer, y=y_train, batch_size=128,
                                 epochs=epochs, verbose=1, callbacks=cbs)

        if early_stopping and os.path.exists(model_checkpoint_path):
            spam_model = NSC.load_saved_model(model_checkpoint_path)

        score = spam_model.evaluate(x=test_input_layer, y=y_test, verbose=1)

        print("Test Score:", score[0])
        print("Test Accuracy:", score[1])

        NSC.plot_model_history(history)

        return spam_model

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

            data = str(status.created_at) + ',' + status.user.name + ',' + status.text.replace('\n', '') + \
                   ',' + self.default_sentiment + '\n'
            print(data)
            f.write(data)
            print('Tweet saved...')


def main(search_past=False, search_stream=False, use_ml=False, phrase='', filter_in=None, filter_out=None,
         history_count=1000):
    if not filter_in:
        filter_in = []
    if not filter_out:
        filter_out = []

    tw = TwitterManager()

    if use_ml:
        tw.initialize_twitter_spam_model()
        # tw.initialize_twitter_sentiment_model()

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


if __name__ == '__main__':

    create_ml_models = True

    main(use_ml=create_ml_models)
