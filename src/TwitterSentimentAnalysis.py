import tweepy
import pandas as pd


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
        tweet_keys = ['Date', 'User', 'Text']

        for tweet in data:

            temp_dict = {}
            temp_dict.update({tweet_keys[0]:tweet.created_at, tweet_keys[1]:tweet.user.name, tweet_keys[2]:tweet.text})

            tweets.append(temp_dict)

        df = pd.DataFrame(data=tweets, columns=tweet_keys)

        return df

    def StartStream(self, phrases, limit=-1):

        # Set filename
        filename = ''
        for phrase in phrases:
            filename += phrase + "_"
        self.listener.output_file = filename + 'tweets_stream.csv'

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


class Utils:

    '''
    Writes a pandas DataFrame to a csv in the current directory
    '''
    @staticmethod
    def WriteDataFrameToCsv(df, filename):

        if df.empty:
            print('No tweets found given criteria. Please try again.')
        else:
            print('Writing tweets to ' + filename + '...')

            # Attempt to write to csv
            try:
                df.to_csv(filename)
            except:
                print('Could not open ' + filename + '. Is the file open?')

    '''
    Constructs a proper advanced twitter search query given certain operations.
    Refer to https://developer.twitter.com/en/docs/twitter-api/v1/rules-and-filtering/overview/standard-operators
    '''
    @staticmethod
    def ConstructQuery(phrase, filter_in=[], filter_out=[], exact_phrase=''):

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


# Entry
if __name__ == "__main__":

    tw = TwitterManager()

    # Search phrase
    phrase = 'vaccine'
    query = Utils.ConstructQuery(phrase, filter_in=['images'], filter_out=['vine', 'retweets'])
    tweets = tw.PhraseSearchHistory(query, 10)
    Utils.WriteDataFrameToCsv(tweets, phrase+'_tweet_history_search.csv')

    # Start query stream
    tw.StartStream(['vaccine'])