import pandas as pd

from TwitterSpamModelInterface import *

"""
1. Grabs Tweets
2. Labels Using Spam Model
3. Evaluates Using Sentiment Model
"""


class OperationHandler:
    def __init__(self, spam_model_path, sentiment_model_path):
        self.spam_model = load_model_from_bin(spam_model_path)
        self.sentiment_model = 0

    def analyze_tweets(self, path):
        """
        Function to analyze a csv of Tweets
        """
        base_df = pd.read_csv(path)


        # Add way to get processed text
        processed_data = self.spam_model.data.get_x_val_from_csv(path)


        labels = self.spam_model.model.predict(processed_data).tolist()
        labels = [max(range(len(y1)), key=y1.__getitem__) for y1 in labels]
        df = pd.read_csv(path)
        df['Model Label'] = labels

        results = self.sentiment_model.evaluate(processed_data, labels)
        df['Sentiment'] = results
        df.to_csv(path, index=False)

