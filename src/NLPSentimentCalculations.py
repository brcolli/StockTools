import nltk
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import pandas as pd
from nltk.tag import pos_tag
from nltk import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import re


"""TdaClientManager

Description:
Module for handling all natural language processing (NLP) calculations. Designed to work on generic classification
problems. Includes methods for data parsing, sanitization, and tagging. Goal is to be able to sanitize text,
classify words and phrases, and split up text strings to token clusters with logical grouping.

Authors: Benjamin Collins
Date: April 22, 2021 
"""


class NLPSentimentCalculations:

    """Handles any function calls related to NLP classifications.
    """

    def __init__(self):

        """Constructor method, downloads necessary NLTK data.
        """

        self.classifier = None
        NLPSentimentCalculations.download_nltk_common()

    @staticmethod
    def download_nltk_common():

        """Downloads specific NLTK data for NLP analysis.
        """

        nltk.download('wordnet')                     # For determining base words
        nltk.download('punkt')                       # Pretrained model to help with tokenizing
        nltk.download('averaged_perceptron_tagger')  # For determining word context

    def train_naivebayes_classifier(self, train_data):

        """Trains a simple Naive Bayes classification model on training data.

        :param train_data: A list of tuples of a dictionary and a string. The dictionary is a map of features to truths.
                           Contains a list of features and classification tags to train a Naive Bayes model
        :type train_data: list(tuple(dict(str-> bool)))
        """
        self.classifier = NaiveBayesClassifier.train(train_data)

    def test_classifier(self, test_data):

        """Tests a classifier on test data.

        :param test_data: A list of tuples of a dictionary and a string. The dictionary is a map of features to truths.
                          Contains a list of features and classification tags to test a model
        :type test_data: list(tuple(dict(str-> bool)))

        :return: The accuracy of the model based on test data
        :rtype: float
        """

        #  TODO include F-score from precision and recall
        accuracy = classify.accuracy(self.classifier, test_data)
        print(f'Accuracy is:{accuracy}')
        print(self.classifier.show_most_informative_features(10))

        return accuracy

    def classify_text(self, text):

        """Classifies text using a model that has been trained. Takes in unclean data and passes it through a
        sanitization function. Clean tokens are then parsed into a dictionary of features and passed to the classifier.

        :param text: Text to be classified
        :type text: str

        :return: The classification tag
        :rtype: str
        """

        custom_tokens = NLPSentimentCalculations.sanitize_text(word_tokenize(text))
        return self.classifier.classify(dict([token, True] for token in custom_tokens))

    @staticmethod
    def get_all_words(all_tokens):

        """Iterates through a list of lists of tokens and returns each token.

        :param all_tokens: A list of lists of tokens
        :type all_tokens: list(list(str))

        :return: One token at a time
        :rtype: str
        """

        for tokens in all_tokens:
            for token in tokens:
                yield token

    @staticmethod
    def get_clean_tokens(all_tokens, stop_words=()):

        """Iterates through a list of tokens and sanitizes each one, returning a list of clean tokens. Sanitizing in
        this context means to remove noise such as stop words, bad characters, and emojis.

        :param all_tokens: A list of lists of tokens
        :type all_tokens: list(list(str))
        :param stop_words: A list of the most common words in English that are typically not useful in analysis
        :type stop_words: list(str)

        :return: A list of lists of sanitized tokens
        :rtype: list(list(str))
        """

        cleaned_tokens = []
        for tokens in all_tokens:
            cleaned_tokens.append(NLPSentimentCalculations.sanitize_text(tokens, stop_words))
        return cleaned_tokens

    @staticmethod
    def get_freq_dist(all_tokens):

        """Gets the frequency distribution of all the words in a set of tokens

        :param all_tokens: A list of lists of tokens
        :type all_tokens: list(list(str))

        :return: A dictionary of word to count pairs
        :rtype: dict(str-> int)
        """

        all_words = NLPSentimentCalculations.get_all_words(all_tokens)
        return FreqDist(all_words)

    @staticmethod
    def get_basic_data_tag(all_tokens):

        """Simple convert of a list of lists of tokens to a mapping of features.

        :param all_tokens: A list of lists of tokens
        :type all_tokens: list(list(str))

        :return: A dictionary of feature mappings
        :rtype: dict(str-> bool)
        """

        for tokens in all_tokens:
            yield dict([token, True] for token in tokens)

    @staticmethod
    def get_basic_dataset(all_tokens, classifier_tag):
        token_tags = NLPSentimentCalculations.get_basic_data_tag(all_tokens)
        return [(class_dict, classifier_tag) for class_dict in token_tags]

    @staticmethod
    def sanitize_text(tweet_tokens, stop_words=()):

        cleaned_tokens = []

        for token, _ in pos_tag(tweet_tokens):

            token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                           '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
            token = re.sub("(@[A-Za-z0-9_]+)", "", token)

            wn = nltk.WordNetLemmatizer()
            token = wn.lemmatize(token)

            if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
                cleaned_tokens.append(token.lower())

        return cleaned_tokens

    # Discards all punctuation as classified by string.punctuation
    @staticmethod
    def remove_punctuation(text):
        return "".join([char for char in text if char not in string.punctuation])

    # Remove non-standard ASCII characters below char value 128
    @staticmethod
    def remove_bad_ascii(text):
        return "".join(i for i in text if ord(i) < 128)

    @staticmethod
    def collect_hashtags(text):
        return re.findall(r'#(\S*)\s?', text)

    @staticmethod
    def compute_tf_idf(tweets):

        # Dummy function to trick sklearn into taking token list
        def dummy_func(doc):
            return doc

        vectorizer = TfidfVectorizer(
            analyzer='word',
            tokenizer=dummy_func,
            preprocessor=dummy_func,
            token_pattern=None
        )

        vectors = vectorizer.fit_transform(tweets)
        feature_names = vectorizer.get_feature_names()

        dense = vectors.todense()
        denselist = dense.tolist()

        # Map TF-IDF results to dictionary
        tf_idf_list = []
        for tweetList in denselist:
            tf_idf_dict = dict.fromkeys(feature_names, 0)
            for i in range(0, len(feature_names)):
                tf_idf_dict[feature_names[i]] = tweetList[i]
            tf_idf_list.append(tf_idf_dict)

        return pd.Series(data=tf_idf_list)


def main():
    nsc = NLPSentimentCalculations()


if __name__ == '__main__':
    main()
