import importlib
import nltk
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from nltk.tag import pos_tag
from nltk import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import re
import math


Utils = importlib.import_module('utilities').Utils


"""NLPSentimentCalculations

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
        :param stop_words: A list of the most common words in English that are typically not useful in analysis;
                           defaults to empty set
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

        """Uses a simple tagging of features to tag and collect data, calls get_basic_data_tag

        :param all_tokens: A list of lists of tokens
        :type all_tokens: list(list(str))
        :param classifier_tag: A classification label to mark all tokens
        :type classifier_tag: str

        :return: A list of tuples of dictionaries of feature mappings to classifiers
        :rtype: list(dict(str-> bool), str)
        """

        token_tags = NLPSentimentCalculations.get_basic_data_tag(all_tokens)
        return [(class_dict, classifier_tag) for class_dict in token_tags]

    @staticmethod
    def sanitize_text(tweet_tokens, stop_words=()):

        """Uses a simple tagging of features to tag and collect data, calls get_basic_data_tag

        :param tweet_tokens: A list of lists of tokens
        :type tweet_tokens: list(list(str))
        :param stop_words: A classification label to mark all tokens; defaults to empty set
        :type stop_words: str

        :return: A list of tuples of dictionaries of feature mappings to classifiers
        :rtype: list(dict(str-> bool), str)
        """

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

    @staticmethod
    def remove_punctuation(text):

        """Discards all punctuation as classified by string.punctuation

        :param text: String text to be cleaned
        :type text: str

        :return: String with removed punctuation
        :rtype: str
        """

        return "".join([char for char in text if char not in string.punctuation])

    @staticmethod
    def remove_bad_ascii(text):

        """Remove non-standard ASCII characters below char value 128

        :param text: String text to be cleaned
        :type text: str

        :return: String with removed bad ascii
        :rtype: str
        """

        return "".join(i for i in text if ord(i) < 128)

    @staticmethod
    def collect_hashtags(text):

        """Collects all hashtags in a string

        :param text: String text to be parsed
        :type text: str

        :return: List of all strings that follow a hashtag symbol
        :rtype: list(str)
        """

        return re.findall(r'#(\S*)\s?', text)

    @staticmethod
    def generate_n_gram(tokens, n):
        return nltk.ngrams(tokens, n)

    @staticmethod
    def compute_tf_idf(train_tokens, test_tokens):

        """Computes Term Frequency-Inverse Document Frequency for a list of text strings. TF-IDF is a numerical
        representation of how important a word is to a document. This weight is proportional to the frequency of the
        term. This vectorizes a list of text, counts the feature name frequencies, and stores them in a dictionary,
        one for each text.

        :param train_tokens: List of tokens from the training set
        :type train_tokens: list(str)
        :param test_tokens: List of tokens from the test set
        :type test_tokens: list(str)

        :return: A pandas series of the frequency of each feature
        :rtype: :class:`pandas.core.series.Series`
        """

        # Dummy function to trick sklearn into taking token list
        def dummy_func(doc):
            return doc

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            decode_error='replace',
            min_df=2,
            analyzer='word',
            tokenizer=dummy_func,
            preprocessor=dummy_func,
            token_pattern=None
        )

        x_train = vectorizer.fit_transform(train_tokens)
        #feature_names = vectorizer.get_feature_names()

        x_test = vectorizer.transform(test_tokens)

        #dense = x_train.todense()
        #denselist = dense.tolist()

        # Map TF-IDF results to dictionary
        '''
        tf_idf_list = []
        for texttList in denselist:
            tf_idf_dict = dict.fromkeys(feature_names, 0)
            for i in range(0, len(feature_names)):
                tf_idf_dict[feature_names[i]] = texttList[i]
            tf_idf_list.append(tf_idf_dict)
        '''

        # TODO maybe filter for just the top 20,000 best features, use sklearn.SelectKBest (will need train labels)

        return x_train, x_test

    @staticmethod
    def show_data_statistics(tokens_class_a, tokens_class_b):

        """Displays various statistics about the given datasets, can be used to analyze and think about the approach.

        :param tokens_class_a: List of data of classification A
        :type tokens_class_a: list(str)
        :param tokens_class_b: List of data of classification B
        :type tokens_class_b: list(str)
        """

        tokens_class_a_count = len(tokens_class_a)
        tokens_class_b_count = len(tokens_class_b)

        dataset = tokens_class_a + tokens_class_b
        dataset_count = len(dataset)

        print(f'Amount of data for classification A: {tokens_class_a_count}; '
              f'{(tokens_class_a_count / dataset_count) * 100}% of total data')
        print(f'Amount of data for classification B: {tokens_class_b_count}; '
              f'{(tokens_class_b_count / dataset_count) * 100}% of total data')

        print(f'Total amount of data: {dataset_count}')

        word_median = np.median([len(s) for s in dataset])
        print(f'Median number of words in sample: {word_median}')

        print(f'amount of data points / median number of words per point: '
              f'{math.ceil(dataset_count / word_median)}')

        # TODO maybe plot some things, like frequency distribution of ngrams


def main():
    nsc = NLPSentimentCalculations()


if __name__ == '__main__':
    main()
