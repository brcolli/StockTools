import pandas as pd

import TwitterSpamModelInterface as SpamMI
import TwitterSentimentModelInterface as SentMI

"""
1. Grabs Tweets from csv
2. Labels Using Spam Model
3. Evaluates Using Sentiment Model
4. Saves outputs to csv
"""


class ModelHandler:
    def __init__(self, spam_model_path):
        self.spam_model = SpamMI.load_model_from_bin(spam_model_path)
        self.sentiment_model = SentMI.load_model()

    def analyze_tweets(self, path):
        """
        Function to analyze a csv of Tweets
        """

        # Read in dataframe from file once
        df = pd.read_csv(path)

        # Predict spam labels: binary output
        labels = self.spam_model.predict(df)

        # Predict sentiment labels: float in (0, 1) where 0 is negative and 1 is positive
        # Format: [[Probability of Negative, P(Positive)], [P(N), P(P)]] where P(N) + P(P) = 1 for each datapoint
        sentiments = self.sentiment_model.raw_predict(df)

        # Grab only the Probability of Positive sentiment to have a single float value
        sentiments = [s[1] for s in sentiments]

        print(labels)
        print(sentiments)

        # Write predictions to dataframe and csv
        df['Model Label'] = labels
        df['Sentiment'] = sentiments
        df.to_csv(path, index=False)

        return df


def main():
    MH = ModelHandler('../data/Learning Data/OH.bin')
    df = MH.analyze_tweets('../data/TweetData/Test0.csv')
    return df


if __name__ == '__main__':
    x = main()
