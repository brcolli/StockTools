import datetime
import tweepy
from bs4 import BeautifulSoup
import requests
import pandas as pd
import nltk
from nltk.corpus import twitter_samples
from utilities import Utils
from NLPSentimentCalculations import NLPSentimentCalculations as nSC
from TwitterModelInterface import TwitterSpamModelInterface as tSPMI
from TwitterModelInterface import TwitterSentimentModelInterface as tSEMI
from typing import List
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from time import sleep


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

    def __init__(self):

        self.driver = webdriver.Chrome(ChromeDriverManager().install())

    def get_linkedin_news(self):

        login_url = 'https://www.linkedin.com/login'
        url = 'https://www.linkedin.com/feed/'

        self.driver.get(login_url)
        sleep(2)

        self.driver.find_element(by=By.ID, value='username').send_keys('sstben@gmail.com')
        self.driver.find_element(by=By.ID, value='password').send_keys('204436Brc!PandC')
        self.driver.find_element(by=By.ID, value='password').send_keys(Keys.RETURN)

        self.driver.get(url)
        sleep(3)
        news_lines = self.driver.find_element(by=By.CLASS_NAME, value="news-module").find_elements(by=By.TAG_NAME, value='li')

        data = {'title': [], 'url': [], 'full_text': []}
        for news in news_lines:

            a = news.find_element(by=By.TAG_NAME, value='a')

            title = a.text.split('\n')[0]
            href = a.get_attribute('href')

            if title != '':
                data['title'].append(title)
            else:
                data['title'].append(' '.join(href.split('/')[-2].split('-')[:-1]))

            data['url'].append(href)

        for href in data['url']:

            r = requests.get(href)
            html = BeautifulSoup(r.text, 'html.parser')

            article = html.find_all('p')

            texts = []
            for paragraph in article:
                texts.append(paragraph.text.replace('\n', '').replace('\t', '').strip())

            data['full_text'].append(' '.join(texts))

        return data

    def get_ticker_news_headlines(self, ticker):

        """Given a ticker, scrapes finviz for news headlines.

        :param ticker: Ticker to scan for news headlines about
        :type ticker: str

        :return: A list of news headlines regarding the ticker
        :rtype: list(str)
        """

        url = 'https://finviz.com/quote.ashx?t=' + ticker

        self.driver.get(url)

        # Parse HTML for headlines
        data = {'title': [], 'url': [], 'full_text': []}
        try:
            news_rows = self.driver.find_element(by=By.ID, value='news-table').find_elements(by=By.TAG_NAME, value='a')
        except Exception as e:
            print(f'Error getting data for {ticker}')
            return None

        for i, row in enumerate(news_rows):

            data['title'].append(row.text)

            href = row.get_attribute('href')
            data['url'].append(href)

        for href in data['url']:

            '''
            self.driver.get(href)
            try:
                article = self.driver.find_elements(by=By.TAG_NAME, value='p')
            except Exception as e:
                print(href)
                print(e)
                continue

            texts = []
            for paragraph in article:
                texts.append(paragraph.text)
            '''

            r = requests.get(href)
            html = BeautifulSoup(r.text, 'html.parser')

            article = html.find_all('p')

            texts = []
            for paragraph in article:
                texts.append(paragraph.text.replace('\n', '').replace('\t', '').strip())

            data['full_text'].append(' '.join(texts))

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

    def write_news_to_csv(self, filename, query):

        data = self.get_ticker_news_headlines(query)
        df = pd.DataFrame(data)
        Utils.write_dataframe_to_csv(df, filename, write_index=False)

    def write_linkedin_news_to_csv(self):

        today = Utils.datetime_to_time_str(datetime.datetime.today())
        filename = f'../data/NewsData/LinkedInNews{today}.csv'

        data = self.get_linkedin_news()
        df = pd.DataFrame(data)
        Utils.write_dataframe_to_csv(df, filename, write_index=False)


class TwitterManager:
    """Handles sentiment analysis for Twitter data, i.e. tweets. Can tag input data or a live stream of tweets.
    """

    def __init__(self):

        """Constructor method, creates a login session for the Twitter API, and sets up a stream listener.
        """

        # Tokens and keys
        # TODO move keys to a hashed file on a cloud server
        self.shKey = 'BUYwpbmEi3A29cm9kOXeX9y8n'
        self.scKey = 'MF5w3g6jmn7WnYM6DG8xtIWkdjnEhInnBSf5bU6HclTF4wSkJ9'
        self.shToken = '4149804506-DrTR0UhuQ8pWf16r9wm8NYdkGNSBWuib2Y8nUlw'
        self.scToken = 'PYkfcY2w6tvovb4RMyCNpEAPyValmReJlaHUgC2KsWWzQ'
        self.bearer = 'AAAAAAAAAAAAAAAAAAAAAHfLGwEAAAAA23M6%2FAOdV1Gp6xMLfN1txD8DUwM%3DlxzfwHEkS1xPDAcAMncrZKVOUJEL0' \
                      'csMBxQCKOFp89CLlcVo6v'

        # Set up authentication
        self.auth = tweepy.OAuthHandler(self.shKey, self.scKey)
        self.auth.set_access_token(self.shToken, self.scToken)

        self.api = tweepy.API(self.auth, wait_on_rate_limit=True)

        # Set up stream listener
        self.listener = TwitterStreamListener()
        self.stream = tweepy.Stream(auth=self.auth, listener=self.listener)

        self.nsc = nSC()

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

        x_train, x_test, y_train, y_test = nSC.keras_preprocessing(dataframe[features_to_train], dataframe['Label'],
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
        x_train_text_clean = [nSC.sanitize_text_string(s) for s in list(x_train_text_data)]
        x_test_text_clean = [nSC.sanitize_text_string(s) for s in list(x_test_text_data)]

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

        tweets_clean = nSC.get_clean_tokens(tweet_tokens)
        return nSC.get_basic_dataset(tweets_clean, classifier_tag)

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

    def phrase_search_history(self, phrase: str, count: int = 1000):

        """Searches historic tweets for a phrase

        TODO handle getting full_text from retweets

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
    def construct_twitter_query(phrase='', filter_in=None, filter_out=None, exact_phrase=''):

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

        if phrase:
            query += phrase + ' OR ' + phrase.upper() + ' OR ' + phrase.capitalize()

        # Add all filtered in requirements
        for fin in filter_in:
            query += ' filter:' + fin

        # Add all filtered out requirements
        for fout in filter_out:
            query += ' -filter:' + fout

        return query

    def start_stream(self, phrases: List[str], default_sentiment: str = '') -> None:

        """Starts a stream to collect tweets based on search phrases

        :param phrases: The phrases to search for in a stream
        :type phrases: list(str)
        :param default_sentiment: The default sentiment to assign to all incoming tweets; defaults to empty
        :type default_sentiment: str
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

    def on_status(self, status: tweepy.api) -> None:

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

            data = str(status.created_at) + ',' + status.user.name + ',' + status.text.replace('\n', '') + ',' +\
                self.default_sentiment + '\n'

            print(data)
            f.write(data)
            print('Tweet saved...')


def generate_metrics_file(queries):

    metric_outputs = '../data/TweetData/sp-100-metrics.csv'

    date_range = '20220801-20220831'

    queries = pd.read_csv(queries, index_col='Symbol')

    metrics = nSC.generate_metrics_output(queries, date_range)

    metrics_df = pd.DataFrame(metrics)

    Utils.write_dataframe_to_csv(metrics_df, metric_outputs, write_index=False)


def main(search_past: bool = False, search_stream: bool = False, train_spam: bool = False, train_sent: bool = False,
         phrase: str = '', filter_in: list = None, filter_out: list = None, history_count: int = 1000, test_file=''):

    """
    :param search_past: Flag for choosing to search past Twitter posts with queries; defaults to False
    :type search_past: Boolean
    :param search_stream: Flag for choosing to search a stream of Twitter data; defaults to False
    :type search_stream: Boolean
    :param train_spam: Flag for choosing to build spam models; defaults to False
    :type train_spam: Boolean
    :param train_sent: Flag for choosing to build sentiment models; defaults to False
    :type train_sent: Boolean
    :param phrase: String to query for; defaults to empty string
    :type phrase: string
    :param filter_in: Types of data to include in query; defaults to []
    :type filter_in: list(str)
    :param filter_out: Types of data to remove from query; defaults to []
    :type filter_out: list(str)
    :param history_count: Number of historical tweets to collect; defaults to 1000
    :type history_count: int
    """

    if not filter_in:
        filter_in = []
    if not filter_out:
        filter_out = []

    tw = TwitterManager()

    spam_model_learning = None
    sentiment_model_learning = None

    test_csv = test_file + '.csv'
    test_df = pd.read_csv(test_csv)

    if train_spam:

        spam_model_learning = tSPMI.create_spam_model_to_train(epochs=5000,
                                                               batch_size=128,
                                                               features_to_train=['full_text'],
                                                               load_to_predict=True,
                                                               checkpoint_model=False,
                                                               model_h5='../data/analysis/Model Results/'
                                                                        'Saved Models/best_spam_model.h5',
                                                               train_data_csv='../data/Learning Data/Spam/'
                                                                              'spam_train_set.csv',
                                                               test_size=0.01)

        spam_score, spam_score_raw = spam_model_learning.predict(test_csv)

        test_df['SpamLabel'] = spam_score
        test_df['SpamConfidence'] = spam_score_raw

    if train_sent:

        sentiment_model_learning = tSEMI.create_sentiment_model_to_train(epochs=5000,
                                                                         batch_size=128,
                                                                         features_to_train=['full_text'],
                                                                         load_to_predict=True,
                                                                         checkpoint_model=False,
                                                                         model_h5='../data/analysis/'
                                                                                  'Model Results/'
                                                                                  'Saved Models/'
                                                                                  'best_sentiment_model.h5',
                                                                         train_data_csv='../data/Learning Data/'
                                                                                        'Sentiment/'
                                                                                        'sentiment_train_set.csv',
                                                                         test_size=0.1)

        sent_score, sent_score_raw = sentiment_model_learning.predict(test_csv)

        test_df['SentimentLabel'] = sent_score
        test_df['SentimentConfidence'] = sent_score_raw

    Utils.write_dataframe_to_csv(test_df, test_file + 'Labeled' + '.csv', write_index=False)

    #if train_spam and train_sent:
        #mh = ModelHandler(spam_model=spam_model_learning, sentiment_model=sentiment_model_learning)
        #mh.analyze_tweets('SOURCE OF TWEETS', out_path='SOURCE TO WRITE TO')

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

    from os import listdir
    from os.path import isfile, join

    q_dir = '../data/TweetData/Historic SP-100_20220901-20221001/'

    beta_companies = pd.read_csv('../doc/beta_companies_keywords.csv')
    beta_queries = beta_companies.columns.to_list()
    for _, row in beta_companies.iterrows():
        for key, val in row.items():
            query = key + ' ' + val
            beta_queries.append(query)

    files = [join(q_dir, f).replace("\\", "/") for f in listdir(q_dir) if isfile(join(q_dir, f)) and 'Scrape' in f]

    queries = []
    for filename in files:
        f = filename.split('/')[-1]
        q = f.split('20220901')[0]

        if q in beta_queries:
            queries.append(filename)

    for f in queries:
        f = f.replace('.csv', '')
        main(train_spam=True, train_sent=True, test_file=f)

    #mdf = nSC.generate_metrics_from_files(queries)

    #Utils.write_dataframe_to_csv(mdf, '../data/TweetData/sp-100-metrics.csv', write_index=False)

    '''
    chosen = pd.read_csv('../doc/chosen_companies.csv')['Name']

    for _, val in chosen.items():

        filename = f'{q_dir}{val}20220901-20221001'

        main(train_spam=True, train_sent=True, test_file=filename)
    '''
