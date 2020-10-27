import importlib
import tweepy
import pandas as pd
import nltk
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords

Utils = importlib.import_module('utilities').Utils
NSC = importlib.import_module('NLPSentimentCalculations').NLPSentimentCalculations


class TwitterManager:

    '''
    Constructor
    '''
    def __init__(self):

        # Tokens and keys
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
        tweets_clean = NSC.get_clean_tokens(tweet_tokens, stopwords.words('english'))
        return NSC.get_basic_dataset(tweets_clean, classifier_tag)

    def get_tweet_sentiment(self, text):
        sentiment = self.nsc.classify_text(text)
        print('Text:', text)
        print('Sentiment:', sentiment)
        return sentiment

    '''
    PhraseSearchHistory
    Searches historic tweets for a phrase
    '''
    def PhraseSearchHistory(self, phrase, count=1000):

        print('Collecting tweets from phrase: ' + phrase + '...')
        data = tweepy.Cursor(self.api.search,
                             q=phrase,
                             lang="en").items(count)

        print('Compiling tweets...')
        tweets = []
        tweet_keys = ['Date', 'User', 'Text', 'Sentiment']

        for tweet in data:

            if self.nsc.classifier is not None:
                sentiment = self.get_tweet_sentiment(tweet.text)
            else:
                sentiment = 'None'

            temp_dict = {}
            temp_dict.update({tweet_keys[0]: tweet.created_at, tweet_keys[1]: tweet.user.name,
                              tweet_keys[2]: tweet.text, tweet_keys[3]: sentiment})

            tweets.append(temp_dict)

        df = pd.DataFrame(data=tweets, columns=tweet_keys)

        return df

    '''
    Constructs a proper advanced twitter search query given certain operations.
    Refer to https://developer.twitter.com/en/docs/twitter-api/v1/rules-and-filtering/overview/standard-operators
    '''
    @staticmethod
    def ConstructTwitterQuery(phrase, filter_in=[], filter_out=[], exact_phrase=''):

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

    def StartStream(self, phrases, limit=-1):

        print('Starting stream on ' + str(phrases))

        # Set filename
        filename = ''
        for phrase in phrases:
            filename += phrase + "_"
        self.listener.output_file = '../data/' + filename + 'tweets_stream.csv'

        self.stream.filter(track=phrases, is_async=True)


class TwitterStreamListener(tweepy.StreamListener):

    def __init__(self):
        super(TwitterStreamListener, self).__init__()
        self.header_written = False
        self.output_file = ''

    def on_status(self, status):

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


def main():

    tw = TwitterManager()

    tw.initialize_nltk_twitter()
    tw.get_tweet_sentiment('Tesla can suck it')

    # Search phrase
    phrase = 'AMD'
    query = tw.ConstructTwitterQuery(phrase, filter_out=['vine', 'retweets', 'links'])
    print(query)
    tweets = tw.PhraseSearchHistory(query, 100)
    Utils.write_dataframe_to_csv(tweets, '../data/' + phrase + '_tweet_history_search.csv')

    # Start query stream
    #tw.StartStream(['$PTON'])


if __name__ == '__main__':
    main()
