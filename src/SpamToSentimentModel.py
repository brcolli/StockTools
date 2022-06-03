import pandas as pd

import TwitterModelInterface as tMI

"""
1. Grabs Tweets from csv
2. Labels Using Spam Model
3. Evaluates Using Sentiment Model
4. Saves outputs to csv
"""


class ModelHandler:

    def __init__(self, spam_model=None, sentiment_model=None, load_spam_model_path=''):
        if spam_model is not None:
            self.spam_model = spam_model
        else:
            self.spam_model = tMI.load_model_from_bin(load_spam_model_path)

        if sentiment_model is not None:
            self.sentiment_model = sentiment_model
        else:
            self.sentiment_model = tMI.load_model()

    def analyze_tweets(self, path, out_path=''):
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

        if out_path:
            write_path = out_path
        else:
            write_path = path

        try:
            df.to_csv(write_path, index=False)
        except FileNotFoundError:
            print(f'Path not found: {write_path}')

        return df


def main():
    mh = ModelHandler('../data/Learning Data/OH.bin')
    df = mh.analyze_tweets('../data/TweetData/Test0.csv')
    return df


if __name__ == '__main__':
    x = main()
