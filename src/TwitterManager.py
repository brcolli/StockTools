import tweepy
import os
from dotenv import load_dotenv


class TwitterManager:

    def __init__(self):

        load_dotenv('../doc/BotometerKeys.env')

        self.twitter_app_auth = {
            'consumer_key': os.getenv('consumer_key'),
            'consumer_secret': os.getenv('consumer_secret')
        }

        self.auth = tweepy.OAuthHandler(self.twitter_app_auth['consumer_key'], self.twitter_app_auth['consumer_secret'])
        self.auth.set_access_token(os.getenv('shToken'), os.getenv('scToken'))
        self.twitter_api = tweepy.API(self.auth, wait_on_rate_limit=True)
