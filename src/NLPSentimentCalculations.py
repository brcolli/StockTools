import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import ToktokTokenizer
import numpy as np
from nltk.tag import pos_tag
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras import backend as kb
import transformers
import string
import re
import math
import utilities
import warnings

Utils = utilities.Utils

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
    def keras_preprocessing(x, y, test_size=0.1, augmented_states=None, remove_bad_labels=True):

        """Categorizes and preprocesses feature and label datasets

        :param x: The input data.
        :type x: list(obj)
        :param y: The output data.
        :type y: list(obj)
        :param test_size: The ratio of test size to the rest of the dataset.
        :type test_size: double
        :param augmented_states: Series containing the augmented flags
        :type augmented_states: pandas.series
        :param remove_bad_labels: Flag to remove labels that aren't normal (i.e. not 0 or 1)
        :type remove_bad_labels: bool

        :return: A tuple of arrays of x and y train and test sets.
        :rtype: tuple(dataframe(obj), dataframe(obj), dataframe(obj), dataframe(obj))
        """

        label_encoder = preprocessing.LabelEncoder()

        # TODO change to TF-IDF?
        if remove_bad_labels:

            bad_rows = y == -1

            y = y[bad_rows == 0]
            x = x[bad_rows == 0]

            if augmented_states is not None:
                augmented_states = augmented_states[bad_rows == 0]

        y = label_encoder.fit_transform(y)

        x_train, x_test, y_train, y_test = NLPSentimentCalculations.split_data_to_train_test(
            x, y, test_size=test_size, augmented_states=augmented_states)

        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)

        return x_train, x_test, y_train, y_test

    def keras_word_embeddings(self, x, word_count=-1):

        """Takes an array of word tokens and converts them into sequence representations. Uses maximum word count to
        select array length. Arrays shorter than word_count will be padded at the end with 0s. Arrays longer than
        word_count will be truncated, keeping most prominent features.

        :param x: The input data.
        :type x: np.array(np.array(str))
        :param word_count: The maximum number of sequences (words) to have in input data. Default -1, will calculate
        the maximum sequence count for you.
        :type word_count: int

        :return: A tuple of maximum sequence count and an array of an array of sequence representations of word tokens.
        :rtype: tuple(int, np.array(np.array(double)))
        """

        x_sequence = self.tokenizer.texts_to_sequences(x)

        if word_count == -1:
            word_count = max([len(s) for s in x_sequence])

        return word_count, tf.keras.preprocessing.sequence.pad_sequences(x_sequence, padding='post', maxlen=word_count)

    def create_glove_word_vectors(self, trained_vector_file='../data/Learning Data/GloVe/glove.6B/glove.6B.100d.txt'):

        """Reads a pre-trained GloVe embedding file and puts them into an embedding matrix, using token word_indices
        to select word weights.

        :param trained_vector_file: Filepath for the GloVe pre-trained embeddings file.
        :type trained_vector_file: str

        :return: A 2D array of pre-trained embedding weights.
        :rtype: np.array(np.array(double))
        """

        embeddings_dict = dict()

        with open(trained_vector_file, encoding='utf8') as gf:

            for line in gf:
                records = line.split()
                word = records[0]
                vector_dimensions = np.asarray(records[1:], dtype='float32')
                embeddings_dict[word] = vector_dimensions

        embedding_matrix = np.zeros((len(self.tokenizer.word_index) + 1, len(embeddings_dict[word])))
        for word, index in self.tokenizer.word_index.items():

            embedding_vector = embeddings_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

        return embedding_matrix

    def create_roberta_tokenizer(self, test_df, sentiment_id, maxlen):

        tokenizer = transformers.ByT5Tokenizer(
            vocab_file='../data/vocab-roberta-base.json',
            merges_file='../data/merges-roberta-base.txt',
            lowercase=True,
            add_prefix_space=True
        )

        ct = test_df.shape[0]
        input_ids_t = np.ones((ct, maxlen), dtype='int32')
        attention_mask_t = np.zeros((ct, maxlen), dtype='int32')
        token_type_ids_t = np.zeros((ct, maxlen), dtype='int32')

        for k in range(test_df.shape[0]):
            # INPUT_IDS
            text1 = " " + " ".join(test_df.loc[k, 'text'].split())
            enc = tokenizer.encode(text1)
            s_tok = sentiment_id[test_df.loc[k, 'sentiment']]
            input_ids_t[k, :len(enc.ids) + 5] = [0] + enc.ids + [2, 2] + [s_tok] + [2]
            attention_mask_t[k, :len(enc.ids) + 5] = 1

        return token_type_ids_t, input_ids_t, attention_mask_t

    def create_sentiment_text_model(self, embedding_matrix, output_shape, maxlen):

        dropout_rate = 0.5

        input_text_layer, lstm_text_layer = self.create_spam_text_submodel(blocks=5,
                                                                           dropout_rate=dropout_rate,
                                                                           filters=64,
                                                                           kernel_size=3,
                                                                           pool_size=2,
                                                                           embedding_matrix=embedding_matrix,
                                                                           maxlen=maxlen,
                                                                           use_cnn=False)

        output_layer = tf.keras.layers.Dense(output_shape[1], activation='softmax')(lstm_text_layer)

        return tf.keras.models.Model(inputs=input_text_layer, outputs=output_layer)

    def create_spam_text_meta_model(self, embedding_matrix, meta_feature_size, output_shape, maxlen):

        """Creates a Tensorflow model that combines a text training model and a meta data training model. If the size
        of the meta features count is 0, will skip the meta model and just return a model for text training.

        The text model can either be a multi-layer perceptron (MLP) or a Separated Convolutional Neural Network
        (SepCNN).

        If the meta model is to be included, uses a simple MLP design.

        The two models are then combined in a concatenation layer and end with a softmax output layer. Since the model
        trains on binary classification, many suggest sigmoid output layer. We went with softmax to ensure output
        probabilities round to 1, and because our label array has 2 columns. May consider going back to sigmoid later.

        :param embedding_matrix: GloVe pre-trained embedding matrix
        :type embedding_matrix: np.array(np.array(double))
        :param meta_feature_size: Number of meta features to train. If 0, will skip meta model.
        :type meta_feature_size: int
        :param output_shape: Shape of the output layer results.
        :type output_shape: tuple(int, int)
        :param maxlen: Maximum number of sequences in the input layer for text training.
        :type maxlen: int

        :return: A Tensorflow model
        :rtype: Tensorflow.model
        """

        dropout_rate = 0.5

        input_text_layer, lstm_text_layer = self.create_spam_text_submodel(blocks=5,
                                                                           dropout_rate=dropout_rate,
                                                                           filters=64,
                                                                           kernel_size=3,
                                                                           pool_size=2,
                                                                           embedding_matrix=embedding_matrix,
                                                                           maxlen=maxlen,
                                                                           use_cnn=False,
                                                                           use_transformers=False)

        if meta_feature_size < 1:
            # No meta data, don't create and concat
            dense_layer = tf.keras.layers.Dense(10, activation='relu')(lstm_text_layer)
            output_layer = tf.keras.layers.Dense(output_shape[1], activation='softmax')(dense_layer)

            return tf.keras.models.Model(inputs=input_text_layer, outputs=output_layer)

        input_meta_layer, dense_meta_layer = NLPSentimentCalculations.create_spam_meta_submodel(meta_feature_size)

        concat_layer = tf.keras.layers.Concatenate()([lstm_text_layer, dense_meta_layer])

        dense_concat = tf.keras.layers.Dense(10, activation='relu')(concat_layer)

        drop = tf.keras.layers.Dropout(rate=dropout_rate)(dense_concat)

        output_layer = tf.keras.layers.Dense(output_shape[1], activation='softmax')(drop)

        return tf.keras.models.Model(inputs=[input_text_layer, input_meta_layer], outputs=output_layer)

    def create_spam_text_submodel(self, blocks, dropout_rate, filters, kernel_size, pool_size, embedding_matrix, maxlen,
                                  use_cnn=False, use_transformers=False):

        """Creates a text-based model for learning. Can take 2 forms. Always begins by creating an Input layer.

        If use_cnn is false, will create an MLP-type model. Creates an embedding layer set to not train, that uses
        the GloVe pre-trained weights. Ends with a Long Short-Term Memory layer.

        If use_cnn is true, will create an embedding layer and chain to a sequence of block layers, count defined by
        input variable blocks. Blocks consist of 2 separable 1D convolutional layers and a MaxPooling layer. Conv blocks
        go into a global average pooling layer.

        Dropout layers are placed throughout model for regularization to reduce overfitting.

        :param blocks: Number of pairs of SepCNN and pooling blocks in the model.
        :type blocks: int
        :param dropout_rate: Percentage of input to drop at Dropout layers.
        :type dropout_rate: double
        :param filters: Output dimension of SepCNN layers in the model.
        :type filters: int
        :param kernel_size: Length of the convolutional window
        :type kernel_size: int
        :param pool_size: Factor by which to downscale input at MaxPooling layer.
        :type pool_size: int
        :param embedding_matrix: GloVe pre-trained embedding matrix
        :type embedding_matrix: np.array(np.array(double))
        :param maxlen: Maximum number of sequences in the input layer for text training.
        :type maxlen: int
        :param use_cnn: Flag to use Separated CNN. If false, will use MLP.
        :type use_cnn: bool
        :param use_transformers: Flag to use transformers. If false, will use MLP.
        :type use_transformers: bool

        :return: A tuple of an input layer and an output layer.
        :rtype: tuple(Tensorflow.layer, Tensorflow.layer)
        """

        text_input_layer = tf.keras.layers.Input(shape=(maxlen,))

        if use_transformers:

            ids = tf.keras.layers.Input((maxlen,), dtype=tf.int32)
            att = tf.keras.layers.Input((maxlen,), dtype=tf.int32)
            tok = tf.keras.layers.Input((maxlen,), dtype=tf.int32)

            config = transformers.RobertaConfig.from_pretrained('../data/config-roberta-base.json')
            bert_model = transformers.TFRobertaModel.from_pretrained('../data/pretrained-roberta-base.h5',
                                                                     config=config)
            x = bert_model(ids, attention_mask=att, token_type_ids=tok)

            x1 = tf.keras.layers.Dropout(0.1)(x[0])
            x1 = tf.keras.layers.Conv1D(1, 1)(x1)
            x1 = tf.keras.layers.Flatten()(x1)
            x1 = tf.keras.layers.Activation('softmax')(x1)

            x2 = tf.keras.layers.Dropout(0.1)(x[0])
            x2 = tf.keras.layers.Conv1D(1, 1)(x2)
            x2 = tf.keras.layers.Flatten()(x2)
            x2 = tf.keras.layers.Activation('softmax')(x2)

            return text_input_layer, tf.keras.layers.LSTM(128)([x1, x2])

        elif use_cnn:

            block_connection = tf.keras.layers.Embedding(input_dim=len(self.tokenizer.word_index) + 1,
                                                         input_length=maxlen,
                                                         output_dim=embedding_matrix.shape[1],
                                                         weights=[embedding_matrix],
                                                         trainable=False)(text_input_layer)

            for _ in range(blocks - 1):
                sep_drop = tf.keras.layers.Dropout(rate=dropout_rate)(block_connection)

                sep_conv1 = tf.keras.layers.SeparableConv1D(filters=filters,
                                                            kernel_size=kernel_size,
                                                            activation='relu',
                                                            bias_initializer='random_uniform',
                                                            depthwise_initializer='random_uniform',
                                                            padding='same')(sep_drop)

                sep_conv2 = tf.keras.layers.SeparableConv1D(filters=filters,
                                                            kernel_size=kernel_size,
                                                            activation='relu',
                                                            bias_initializer='random_uniform',
                                                            depthwise_initializer='random_uniform',
                                                            padding='same')(sep_conv1)

                block_connection = tf.keras.layers.MaxPooling1D(pool_size=pool_size)(sep_conv2)

            sep_conv1 = tf.keras.layers.SeparableConv1D(filters=filters * 2,
                                                        kernel_size=kernel_size,
                                                        activation='relu',
                                                        bias_initializer='random_uniform',
                                                        depthwise_initializer='random_uniform',
                                                        padding='same')(block_connection)

            sep_conv2 = tf.keras.layers.SeparableConv1D(filters=filters * 2,
                                                        kernel_size=kernel_size,
                                                        activation='relu',
                                                        bias_initializer='random_uniform',
                                                        depthwise_initializer='random_uniform',
                                                        padding='same')(sep_conv1)

            ga_pooling = tf.keras.layers.GlobalAveragePooling1D()(sep_conv2)

            return text_input_layer, tf.keras.layers.Dropout(rate=dropout_rate)(ga_pooling)

        else:

            embedding_layer = tf.keras.layers.Embedding(input_dim=len(self.tokenizer.word_index) + 1,
                                                        input_length=maxlen,
                                                        output_dim=embedding_matrix.shape[1],
                                                        weights=[embedding_matrix],
                                                        trainable=False)(text_input_layer)

            drop = tf.keras.layers.Dropout(rate=dropout_rate)(embedding_layer)

            return text_input_layer, tf.keras.layers.LSTM(128)(drop)


    @staticmethod
    def create_spam_meta_submodel(meta_feature_size):

        """Creates a model based on training with meta features, non-textual features. If meta_feature_size is 0,
        returns None for both input and output layers.

        Creates a simple MLP consisting of dense ReLU layers.

        :param meta_feature_size: Number of meta features to train. If 0, will skip meta model.
        :type meta_feature_size: int

        :return: A tuple of an input layer and an output layer.
        :rtype: tuple(Tensorflow.layer, Tensorflow.layer)
        """

        if meta_feature_size < 1:
            return None, None

        meta_input_layer = tf.keras.layers.Input(shape=(meta_feature_size,))
        dense_layer_1 = tf.keras.layers.Dense(100, activation='relu')(meta_input_layer)

        return meta_input_layer, tf.keras.layers.Dense(10, activation='relu')(dense_layer_1)

    @staticmethod
    def create_early_stopping_callback(monitor_stat, monitor_mode='auto', patience=0, min_delta=0):

        """Creates a callback for a Tensorflow object, that will trigger early stopping of a model based on conditions.
        Can set the metric to monitor, the amount of patience, and the minimum change required.

        As the model learns, the callback will look at the progress of the metric that is being monitored. If the
        model doesn't improve by min_delta in patience number of epochs, will exit the model early.

        :param monitor_stat: Metric to monitor performance of.
        :type monitor_stat: str
        :param monitor_mode: Type of monitoring. Default is 'auto'. In 'auto', direction of improvement is inferred
        by the name of the metric. In 'min', direction of improvement is decreasing the metric. In 'max', direction of
        improvement is increasing the metric.
        :type monitor_mode: str
        :param patience: Number of epochs to wait for metric improvement.
        :type patience: int
        :param min_delta: Size difference for the metric to be considered improving.
        :type min_delta: double

        :return: A callback for early stopping.
        :rtype: Tensorflow.callback
        """

        return tf.keras.callbacks.EarlyStopping(monitor=monitor_stat, mode=monitor_mode, verbose=1,
                                                patience=patience, min_delta=min_delta)

    @staticmethod
    def create_model_checkpoint_callback(filepath, monitor_stat, mode='auto'):

        """Creates a callback for a Tensorflow object, that will trigger early stopping of a model based on conditions.
        Can set the metric to monitor, the amount of patience, and the minimum change required.

        As the model learns, the callback will look at the progress of the metric that is being monitored. If the
        model doesn't improve by min_delta in patience number of epochs, will exit the model early.

        :param filepath: File location to save the model checkpoint to.
        :type filepath: str
        :param monitor_stat: Metric to monitor performance of.
        :type monitor_stat: str
        :param mode: Type of monitoring. Default is 'auto'. In 'auto', direction of improvement is inferred
        by the name of the metric. In 'min', direction of improvement is decreasing the metric. In 'max', direction of
        improvement is increasing the metric.
        :type mode: str

        :return: A callback for model checkpoints.
        :rtype: Tensorflow.callback
        """

        return tf.keras.callbacks.ModelCheckpoint(filepath, monitor=monitor_stat, mode=mode, verbose=1,
                                                  save_best_only=True)

    @staticmethod
    def load_saved_model(filepath):

        """Loads a model from a model checkpoint file. Will be uncompiled, need to compile model before evaluating.

        :param filepath: File location to load the model checkpoint from.
        :type filepath: str

        :return: A saved model
        :rtype: Tensorflow.model
        """
        return tf.keras.models.load_model(filepath, compile=False)

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

        custom_tokens = NLPSentimentCalculations.sanitize_text_tokens(ToktokTokenizer().tokenize(text))
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
            cleaned_tokens.append(NLPSentimentCalculations.sanitize_text_tokens(tokens))
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
    def get_wordnet_pos(word):

        """Map POS tag to first character lemmatize() accepts"""

        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    @staticmethod
    def sanitize_text_string(sen):

        sentence = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_.&+#]|[!*\(\),]|' \
                          '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', sen)

        sentence = re.sub("(@[A-Za-z0-9_]+)", "", sentence)

        sentence = re.sub('[^a-zA-Z]', ' ', sentence)

        sentence = re.sub(r'\s+', ' ', sentence)

        sentence = ToktokTokenizer().tokenize(sentence.lower())

        stop_words = stopwords.words('english')
        sentence = [word for word in sentence if word not in stop_words and word not in string.punctuation]

        wn = nltk.WordNetLemmatizer()
        sentence = [wn.lemmatize(word, NLPSentimentCalculations.get_wordnet_pos(word)) for word in sentence]

        if len(sentence) > 0:
            return ' '.join(sentence)
        else:
            return ''

    @staticmethod
    def sanitize_text_tokens(tweet_tokens):

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

            stop_words = stopwords.words('english')

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
    def split_data_to_train_test(x, y, test_size=0.1, random_state=11, augmented_states=None):

        """Splits data into randomized train and test subsets.

        :param x: The input data.
        :type x: list(obj)
        :param y: The output data.
        :type y: list(obj)
        :param test_size: The size of the test dataset, between 0.0 and 1.0, as a fractional portion of the train size.
        :type test_size: float
        :param random_state: Randomization seed
        :type random_state: int
        :param augmented_states: Series containing the augmented flags
        :type augmented_states: pandas.series

        :return: A tuple of arrays of x and y train and test sets.
        :rtype: tuple(list(obj), list(obj), list(obj), list(obj))
        """

        if augmented_states is not None:
            augmented_states = list(augmented_states)
            max_test_size = augmented_states.count(0) / len(augmented_states)
            if max_test_size < test_size:
                warnings.warn(f"Too much augmented data, impossible to maintain test size of {test_size} Your max of: "
                              f"{max_test_size} will be used", UserWarning)

                test_size = max_test_size - 0.001

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
    def check_units(y_true, y_pred):

        """Checks the shape of the classification labels and reshapes to 1D as needed

        :param y_true: The actual classification labels.
        :type y_true: np.array()
        :param y_pred: The predicted classification labels.
        :type y_pred: np.array()

        :return: Properly shaped classification labels
        :rtype: tuple(np.array(), np.array())
        """

        if y_pred.shape[1] != 1:
            y_pred = y_pred[:, 1:2]
            y_true = y_true[:, 1:2]

        return y_true, y_pred

    @staticmethod
    @tf.function
    def precision(y_true, y_pred):

        """Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.

        :param y_true: The actual classification labels.
        :type y_true: np.array()
        :param y_pred: The predicted classification labels.
        :type y_pred: np.array()

        :return: Global precision score
        :rtype: double
        """

        y_true, y_pred = NLPSentimentCalculations.check_units(y_true, y_pred)

        true_positives = kb.sum(kb.round(kb.clip(y_true * y_pred, 0, 1)))
        predicted_positives = kb.sum(kb.round(kb.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + kb.epsilon())

        return precision

    @staticmethod
    @tf.function
    def recall(y_true, y_pred):

        """Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.

        :param y_true: The actual classification labels.
        :type y_true: np.array()
        :param y_pred: The predicted classification labels.
        :type y_pred: np.array()

        :return: Global recall score
        :rtype: double
        """

        y_true, y_pred = NLPSentimentCalculations.check_units(y_true, y_pred)

        true_positives = kb.sum(kb.round(kb.clip(y_true * y_pred, 0, 1)))
        possible_positives = kb.sum(kb.round(kb.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + kb.epsilon())

        return recall

    @staticmethod
    @tf.function
    def mcor(y_true, y_pred):

        """Computes the Matthew Correlation Coefficient, the measure of quality of binary classifications.
        Correlation between the observed and predicted binary labels, double between -1 and 1.
        -1 means total disagreement between true and predicted.
        0 means equal to random prediction.
        1 means a perfect relation between true and predicted.

        :param y_true: The actual classification labels.
        :type y_true: np.array()
        :param y_pred: The predicted classification labels.
        :type y_pred: np.array()

        :return: Correlation coefficient
        :rtype: double
        """

        y_true, y_pred = NLPSentimentCalculations.check_units(y_true, y_pred)

        # matthews_correlation
        y_pred_pos = kb.round(kb.clip(y_pred, 0, 1))
        y_pred_neg = 1 - y_pred_pos

        y_pos = kb.round(kb.clip(y_true, 0, 1))
        y_neg = 1 - y_pos

        tp = kb.sum(y_pos * y_pred_pos)
        tn = kb.sum(y_neg * y_pred_neg)

        fp = kb.sum(y_neg * y_pred_pos)
        fn = kb.sum(y_pos * y_pred_neg)

        numerator = (tp * tn - fp * fn)
        denominator = kb.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        return numerator / (denominator + kb.epsilon())

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

        history_params = []
        plt.figure(1)
        for key in history.history.keys():
            if key != 'loss':
                history_params.append(key)
                plt.plot(history.history[key])

        plt.title('model scores')
        plt.ylabel('scores')
        plt.xlabel('epoch')
        plt.legend(history_params, loc='upper left')

        plt.figure(2)
        plt.plot(history.history['loss'])

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


def main():
    nsc = NLPSentimentCalculations()


if __name__ == '__main__':
    main()
