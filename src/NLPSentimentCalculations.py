import importlib
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import numpy as np
from nltk.tag import pos_tag
from nltk import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
import string
import re
import math
import operator

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

        self.tokenizer = tf.keras.preprocessing.text.Tokenizer()

    @staticmethod
    def download_nltk_common():

        """Downloads specific NLTK data for NLP analysis.
        """

        nltk.download('wordnet')  # For determining base words
        nltk.download('punkt')  # Pretrained model to help with tokenizing
        nltk.download('averaged_perceptron_tagger')  # For determining word context

    def train_naivebayes_classifier(self, train_data):

        """Trains a simple Naive Bayes classification model on training data.

        :param train_data: A list of tuples of a dictionary and a string. The dictionary is a map of features to truths.
                           Contains a list of features and classification tags to train a Naive Bayes model
        :type train_data: list(tuple(dict(str-> bool)), str)
        """
        self.classifier = NaiveBayesClassifier.train(train_data)

    @staticmethod
    def keras_preprocessing(x, y, augmented_states=None):

        """Categorizes and preprocesses feature and label datasets

        :param x: The input data.
        :type x: list(obj)
        :param y: The output data.
        :type y: list(obj)

        :return: A tuple of arrays of x and y train and test sets.
        :rtype: tuple(dataframe(obj), dataframe(obj), dataframe(obj), dataframe(obj))
        """

        label_encoder = preprocessing.LabelEncoder()

        # TODO change to TF-IDF?
        y = label_encoder.fit_transform(y)

        x_train, x_test, y_train, y_test = NLPSentimentCalculations.split_data_to_train_test(x, y,
                                            augmented_states=augmented_states)

        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)

        return x_train, x_test, y_train, y_test

    def keras_word_embeddings(self, x, maxlen=300):

        x_sequence = self.tokenizer.texts_to_sequences(x)
        return tf.keras.preprocessing.sequence.pad_sequences(x_sequence, padding='post', maxlen=maxlen)

    def create_glove_word_vectors(self, trained_vector_file='../data/Learning Data/GloVe/glove.6B/glove.6B.100d.txt'):

        embeddings_dict = dict()

        with open(trained_vector_file, encoding='utf8') as gf:

            for line in gf:
                records = line.split()
                word = records[0]
                vector_dimensions = np.asarray(records[1:], dtype='float32')
                embeddings_dict[word] = vector_dimensions

        embedding_matrix = np.zeros((len(self.tokenizer.word_index) + 1, 100))
        for word, index in self.tokenizer.word_index.items():

            embedding_vector = embeddings_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

        return embedding_matrix

    def create_text_meta_model(self, embedding_matrix, meta_feature_size, maxlen=300):

        input_text_layer, lstm_text_layer = self.create_text_submodel(embedding_matrix, maxlen)
        input_meta_layer, dense_meta_layer = NLPSentimentCalculations.create_meta_submodel(meta_feature_size)

        concat_layer = tf.keras.layers.Concatenate()([lstm_text_layer, dense_meta_layer])

        dense_concat = tf.keras.layers.Dense(10, activation='relu')(concat_layer)

        output_layer = tf.keras.layers.Dense(10, activation='softmax')(dense_concat)

        return tf.keras.models.Model(inputs=[input_text_layer, input_meta_layer], outputs=output_layer)

    def create_text_submodel(self, embedding_matrix, maxlen=300):

        text_input_layer = tf.keras.layers.Input(shape=(maxlen,))

        embedding_layer = tf.keras.layers.Embedding(len(self.tokenizer.word_index) + 1,
                                                    100, weights=[embedding_matrix], trainable=False)(text_input_layer)

        return text_input_layer, tf.keras.layers.LSTM(128)(embedding_layer)

    @staticmethod
    def create_meta_submodel(meta_feature_size):

        meta_input_layer = tf.keras.layers.Input(shape=(meta_feature_size,))

        dense_layer_1 = tf.keras.layers.Dense(10, activation='relu')(meta_input_layer)
        return meta_input_layer, tf.keras.layers.Dense(10, activation='relu')(dense_layer_1)

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

    @staticmethod
    def tokenize_string(text):

        """Tokenizes a string, to unigrams.

        :param text: Text to be tokenized
        :type text: str

        :return: The tokens from the text
        :rtype: list(str)
        """

        return word_tokenize(text)

    def classify_text(self, text):

        """Classifies text using a model that has been trained. Takes in unclean data and passes it through a
        sanitization function. Clean tokens are then parsed into a dictionary of features and passed to the classifier.

        :param text: Text to be classified
        :type text: str

        :return: The classification tag
        :rtype: str
        """

        custom_tokens = NLPSentimentCalculations.sanitize_text_tokens(word_tokenize(text))
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
            cleaned_tokens.append(NLPSentimentCalculations.sanitize_text_tokens(tokens, stop_words))
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
    def sanitize_text_string(sen, stop_words=()):

        sentence = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                          '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', sen)

        sentence = re.sub('[^a-zA-Z]', ' ', sentence)

        sentence = re.sub(r'\s+', ' ', sentence)

        sentence = re.sub("(@[A-Za-z0-9_]+)", "", sentence)

        wn = nltk.WordNetLemmatizer()
        sentence = wn.lemmatize(sentence)

        if len(sentence) > 0 and sentence not in string.punctuation and sentence.lower() not in stop_words:
            return sentence.lower()
        else:
            return ''

    @staticmethod
    def sanitize_text_tokens(tweet_tokens, stop_words=()):

        """Cleans text data by removing bad punctuation, emojies, and lematizes.

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
    def generate_n_grams(tokens, n):

        """Creates n-grams from a list of tokens. An n-gram is an N pairing of adjacent strings in text.

        :param tokens: List of tokens to create n-grams from.
        :type tokens: list(str)
        :param n: The gram value to create.
        :type n: int

        :return: Returns an n-gram list.
        :rtype: list(str)
        """

        return [' '.join(grams) for grams in nltk.ngrams(tokens, n)]

    @staticmethod
    def split_data_to_train_test(x, y, test_size=0.3, random_state=11, augmented_states=None):

        """Splits data into randomized train and test subsets.

        :param x: The input data.
        :type x: list(obj)
        :param y: The output data.
        :type y: list(obj)
        :param test_size: The size of the test dataset, between 0.0 and 1.0, as a fractional portion of the train size.
        :type test_size: float
        :param random_state: Randomization seed
        :type random_state: int

        :return: A tuple of arrays of x and y train and test sets.
        :rtype: tuple(list(obj), list(obj), list(obj), list(obj))
        """
        if augmented_states is not None:
            max_test_size = list(augmented_states).count(0) / len(augmented_states)
            if max_test_size < test_size:
                print(f"Too much augmented data, impossible to maintain a test size of {test_size}")
                print(f"Max test size: {round(max_test_size, 2)}")
                return False, False, False, False
            else:
                non_aug = [i for i in range(len(augmented_states)) if augmented_states[i] == 0]
                aug = list(set(non_aug) ^ set(list(range(len(augmented_states)))))
                aug.sort()
                x_a = x.iloc[aug]
                x = x.iloc[non_aug]
                y_a = [y[i] for i in aug]
                y = [y[i] for i in non_aug]

                true_ts = test_size * len(augmented_states) / len(non_aug)

                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=true_ts, random_state=random_state)
                x_train = pd.concat([x_train, x_a])
                y_train = y_train + y_a

                return x_train, x_test, y_train, y_test
        else:
            return train_test_split(x, y, test_size=test_size, random_state=random_state)

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
        # feature_names = vectorizer.get_feature_names()

        x_test = vectorizer.transform(test_tokens)

        # dense = x_train.todense()
        # denselist = dense.tolist()

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

    @staticmethod
    def plot_model_history(history):

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])

        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


def main():
    nsc = NLPSentimentCalculations()


if __name__ == '__main__':
    main()
