import tweepy
from bs4 import BeautifulSoup
import requests
import pandas as pd
import nltk
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
import pickle
import tensorflow as tf
import os
import NLPSentimentCalculations
import time
import utilities

Utils = utilities.Utils

NSC = NLPSentimentCalculations.NLPSentimentCalculations


class ModelParameters:
    """Idea of this class is to be called to check on the version of the model and get info about the model's training.
    """
    
    def __init__(self,
                 comp_epochs=0,
                 saved_model_bin='',
                 original_data_csv='',
                 augmented_data_csv='',
                 early_stopping=False,
                 early_stopping_patience=0,
                 batch_size=128,
                 test_size=0.3,
                 features_to_train=None):

        if features_to_train is None:
            self.features_to_train = ['full_text']

        self.comp_epochs = comp_epochs
        self.saved_model_bin = saved_model_bin
        self.original_data_csv = original_data_csv
        self.augmented_data_csv = augmented_data_csv
        if augmented_data_csv:
            self.has_augmented = True
        else:
            self.has_augmented = False
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.batch_size = batch_size
        self.test_size = test_size
        self.accuracy = 0
        self.test_score = 0
        self.precision = 0
        self.recall = 0
        self.f_score = 0


class Data:
    """This single class should represent all the data being used in the model. Point is that functions to load and process
    data are contained within the class. In case new data is passed, the class processes it. Otherwise, the class should
    load itself from a pickle.
    """
    
    def __init__(self, nsc, base_data_csv, test_size, features_to_train, aug_data_csv=None, from_preload_bin=''):

        self.nsc = nsc
        self.base_data_csv = base_data_csv
        self.test_size = test_size
        self.features_to_train = features_to_train
        self.aug_data_csv = aug_data_csv

        self.x_train_text_embeddings = None
        self.x_test_text_embeddings = None
        self.x_train_meta = None
        self.x_test_meta = None
        self.glove_embedding_matrix = None
        self.y_train = None
        self.y_test = None

        if from_preload_bin:
            self.load_data_from_bin(from_preload_bin)

        else:
            self.load_data_from_csv()

    def get_dataset_from_tweet_spam(self, dataframe, features_to_train=None, test_size=None):

        """Converts the text feature to a dataset of labeled unigrams and bigrams.

        :param dataframe: A dataframe containing the text key with all the text features to parse
        :type dataframe: :class:`pandas.core.frame.DataFrame`
        :param features_to_train: The list of all features to train on, does not need to include 'Label'
        :type features_to_train: list(str)

        :return: A list of tuples of dictionaries of feature mappings to classifiers
        :rtype: list(dict(str-> bool), str)
        """

        if not features_to_train:
            features_to_train = ['full_text']

        x_train, x_test, y_train, y_test = NSC.keras_preprocessing(dataframe[features_to_train], dataframe['Label'],
                                                                   augmented_states=dataframe['augmented'],
                                                                   test_size=test_size)

        if x_train is False:
            print("Train test failed due to over augmentation")
            return False

        # Split into text and meta data

        if 'full_text' in features_to_train:
            x_train_text_data = x_train['full_text']
            x_test_text_data = x_test['full_text']

            features_to_train.remove('full_text')
        else:
            x_train_text_data = pd.DataFrame()
            x_test_text_data = pd.DataFrame()

        # Clean the textual data
        x_train_text_clean = [NSC.sanitize_text_string(s) for s in list(x_train_text_data)]
        x_test_text_clean = [NSC.sanitize_text_string(s) for s in list(x_test_text_data)]

        # Initialize tokenizer on training data
        self.nsc.tokenizer.fit_on_texts(x_train_text_clean)

        # Create word vectors from tokens
        x_train_text_embeddings = self.nsc.keras_word_embeddings(x_train_text_clean)
        x_test_text_embeddings = self.nsc.keras_word_embeddings(x_test_text_clean)

        glove_embedding_matrix = self.nsc.create_glove_word_vectors()

        x_train_meta = x_train[features_to_train]
        x_test_meta = x_test[features_to_train]

        return x_train_text_embeddings, x_test_text_embeddings, x_train_meta, x_test_meta, \
               glove_embedding_matrix, y_train, y_test

    def load_data_from_csv(self):
        twitter_df = Utils.parse_json_botometer_data(self.base_data_csv, self.features_to_train)
        if 'augmented' not in twitter_df.columns:
            twitter_df['augmented'] = 0

        if self.aug_data_csv:
            aug_df = Utils.parse_json_botometer_data(self.aug_data_csv, self.features_to_train)
            if 'augmented' not in aug_df.columns:
                aug_df['augmented'] = 1

            twitter_df = pd.concat([twitter_df, aug_df])

        self.x_train_text_embeddings, self.x_test_text_embeddings, self.x_train_meta, self.x_test_meta, \
        self.glove_embedding_matrix, self.y_train, self.y_test = self.get_dataset_from_tweet_spam(twitter_df,
                                                                self.features_to_train, self.test_size)

    def load_data_from_bin(self, from_preprocess_binary):
        if os.path.exists(from_preprocess_binary):
            with open(from_preprocess_binary, "rb") as fpb:
                data = pickle.load(fpb)

            self.x_train_text_embeddings, self.x_test_text_embeddings, self.x_train_meta, self.x_test_meta, \
            self.glove_embedding_matrix, self.y_train, self.y_test, self.nsc.tokenizer = data

    def save_data_to_bin(self):
        data = (self.x_train_text_embeddings, self.x_test_text_embeddings, self.x_train_meta, self.x_test_meta,
                self.glove_embedding_matrix, self.y_train, self.y_test, self.nsc.tokenizer)

    def save_data(self):
        pass


# 
class Model:
    """This is the main Model class. It will create objects from Data() and Version() classes. Added functionality will be
    to load the entire model in from a file, get version info on previous models, reuse data classes in case the same data
    is to be used with different hyperparamaters, and to add model testing / operation functionality.
    """
    
    def __init__(self):
        self.model = tf.keras.models.Model
        self.version = Version
        self.data = Data
        self.test_dataset = None
        self.nsc = NSC()

    # TODO: Add function to test the model on provided csv of Tweets. Would be useful for validation later.
    # TODO: Add functions to calculate, store, and return aspects apart from accuracy like fscore, precision, and recall

    def get_dataset_from_tweet_spam(self, dataframe, features_to_train=None, test_size=None):

        """Converts the text feature to a dataset of labeled unigrams and bigrams.

        :param dataframe: A dataframe containing the text key with all the text features to parse
        :type dataframe: :class:`pandas.core.frame.DataFrame`
        :param features_to_train: The list of all features to train on, does not need to include 'Label'
        :type features_to_train: list(str)

        :return: A list of tuples of dictionaries of feature mappings to classifiers
        :rtype: list(dict(str-> bool), str)
        """

        if not features_to_train:
            features_to_train = ['full_text']

        # if 'augmented' not in dataframe.keys():
        #     dataframe['augmented'] = 0
        #
        # if augmented_df is not None:
        #     if 'augmented' not in augmented_df.keys():
        #         augmented_df['augmented'] = 1
        #     dataframe = pd.concat([dataframe, augmented_df])

        x_train, x_test, y_train, y_test = NSC.keras_preprocessing(dataframe[features_to_train], dataframe['Label'],
                                                                   augmented_states=dataframe['augmented'],
                                                                   test_size=test_size)

        if x_train is False:
            print("Train test failed due to over augmentation")
            return False

        # Split into text and meta data

        if 'full_text' in features_to_train:
            x_train_text_data = x_train['full_text']
            x_test_text_data = x_test['full_text']

            features_to_train.remove('full_text')
        else:
            x_train_text_data = pd.DataFrame()
            x_test_text_data = pd.DataFrame()

        # Clean the textual data
        x_train_text_clean = [NSC.sanitize_text_string(s) for s in list(x_train_text_data)]
        x_test_text_clean = [NSC.sanitize_text_string(s) for s in list(x_test_text_data)]

        # Initialize tokenizer on training data
        self.nsc.tokenizer.fit_on_texts(x_train_text_clean)

        # Create word vectors from tokens
        x_train_text_embeddings = self.nsc.keras_word_embeddings(x_train_text_clean)
        x_test_text_embeddings = self.nsc.keras_word_embeddings(x_test_text_clean)

        glove_embedding_matrix = self.nsc.create_glove_word_vectors()

        x_train_meta = x_train[features_to_train]
        x_test_meta = x_test[features_to_train]

        return x_train_text_embeddings, x_test_text_embeddings, x_train_meta, x_test_meta, \
               glove_embedding_matrix, y_train, y_test

    def load_data_from_csv(self, base_data_csv, test_size, features_to_train, aug_data_csv=None, to_preload_bin=None):
        twitter_df = Utils.parse_json_botometer_data(base_data_csv, features_to_train)
        if 'augmented' not in twitter_df.columns:
            twitter_df['augmented'] = 0

        if aug_data_csv:
            aug_df = Utils.parse_json_botometer_data(aug_data_csv, features_to_train)
            if 'augmented' not in aug_df.columns:
                aug_df['augmented'] = 1

            twitter_df = pd.concat([twitter_df, aug_df])

        x_train_text_embeddings, x_test_text_embeddings, x_train_meta, x_test_meta, \
        glove_embedding_matrix, y_train, y_test = self.get_dataset_from_tweet_spam(twitter_df, features_to_train,
                                                                                   test_size)

        self.data = Data(x_train_text_embeddings, x_test_text_embeddings, x_train_meta, x_test_meta,
                         glove_embedding_matrix, y_train, y_test, aug_data_csv, base_data_csv, to_preload_bin, test_size)



    def build_model_from_data(self, x_train_text_embeddings, x_test_text_embeddings, x_train_meta, x_test_meta,
                              glove_embedding_matrix, y_train, y_test, epochs, batch_size, saved_model_bin,
                              early_stopping, early_stopping_patience):

        spam_model = self.nsc.create_text_meta_model(glove_embedding_matrix,
                                                     len(x_train_meta.columns), len(x_train_text_embeddings[0]))

        # Print model summary
        spam_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        print(spam_model.summary())

        cbs = []
        if early_stopping:
            # Set up early stopping callback
            cbs.append(NSC.create_early_stopping_callback('acc', patience=early_stopping_patience))

            cbs.append(NSC.create_model_checkpoint_callback(saved_model_bin, monitor_stat='acc'))

        if len(x_train_meta.columns) < 1:
            train_input_layer = x_train_text_embeddings
            test_input_layer = x_test_text_embeddings
        else:
            train_input_layer = [x_train_text_embeddings, x_train_meta]
            test_input_layer = [x_test_text_embeddings, x_test_meta]

        history = spam_model.fit(x=train_input_layer, y=y_train, batch_size=batch_size,
                                 epochs=epochs, verbose=1, callbacks=cbs)

        if early_stopping and os.path.exists(saved_model_bin):
            spam_model = NSC.load_saved_model(saved_model_bin)

        score = spam_model.evaluate(x=test_input_layer, y=y_test, verbose=1, callbacks=[])

        print("Test Score:", score[0])
        print("Test Accuracy:", score[1])

        NSC.plot_model_history(history)

        self.model = spam_model

        return self.model, score
