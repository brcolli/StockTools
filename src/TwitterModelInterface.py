import os.path
import dill
import json
import pandas as pd
import TweetDatabaseManager
from TweetDatabaseManager import TweetDatabaseManager
import ModelBase
import TwitterSpamModel
import TwitterSentimentModel
from utilities import Utils


"""TwitterModelInterface

Description:
Module to work with the TwitterModel.py class with functions to create a new model from data, preprocess data, save
a model for future predictions, or load a saved model.

This module needs to be separate from TwitterSpamModel.py so that the dill library
(saving library) works properly. 

Authors: Fedya Semenov, Benjamin Collins
Creation date: October 13, 2021
"""


class TwitterModelInterface:

    @staticmethod
    def get_settings_dict(json_settings='../data/Learning Data/spam_settings.json', learning_rate=1e-3,
                          epochs=1000, early_stopping=False, checkpoint_model=False, early_stopping_patience=0,
                          batch_size=128, evaluate_model=True, debug=False,
                          train_data_csv='../data/Learning Data/spam_train.csv',
                          aug_data_csv='../data/Learning Data/spam_train_aug.csv',
                          test_size=0.1, preload_train_data_dill='', save_train_data_dill='',
                          model_h5='../data/Learning Data/best_spam_model.h5') -> dict:
        """
        Function that processes all spam model args (both passed and from json) and returns one dictionary of args

        :param json_settings: An optional path to a json settings file which contains a dictionary of values that are
                                treated as args passed to this function
        :type json_settings: str
        :param learning_rate: Learning rate of model
        :type learning_rate: float
        :param epochs: Number of epochs to train the model on
        :type epochs: int
        :param early_stopping: Whether or not to use Early Stopping during model training
        :type early_stopping: bool
        :param checkpoint_model: Whether or not to save model checkpoints during training
        :type checkpoint_model: bool
        :param early_stopping_patience: In the case of early stopping, number of epochs to run without improvement
                                        before stopping the training
        :type early_stopping_patience: int
        :param batch_size: Data batch size to train the model with
        :type batch_size: int
        :param evaluate_model: Whether or not to "evaluate" the model performance after training on test dataset
        :type evaluate_model: bool
        :param debug: Whether or not to run the model in debugging mode
        :type debug: bool
        :param train_data_csv: Path to the train data csv file containing unaugmented training set
                                (can contain augmented rows if 'augment' column is present)
        :type train_data_csv: str
        :param aug_data_csv: Path to augmented train data csv file (containing augmented Tweet dataframe in csv format)
        :type aug_data_csv: str
        :param test_size: The test size (or portion of data to be used for test set during training)
        :type test_size: float 0 < x < 1
        :param preload_train_data_dill: Path to a dill file containing preprocessed training data (used in place of
                                        loading from CSVs)
        :type preload_train_data_dill: str
        :param save_train_data_dill: Path to save training data to a dill file after it gets processed (for future
                                     loading)
        :type save_train_data_dill: str
        :param model_h5: Path to save trained model in h5 format to after training
        :type model_h5: str

        :return: Dictionary of spam model settings that can be used to initialize SpamModelParameters
        :rtype: dict
        """

        # Most values come from passed arguments
        # Some values are pre-set because they never need to be changed in this mode of loading
        settings_dict = {
            'learning_rate': learning_rate,
            'epochs': epochs,
            'early_stopping': early_stopping,
            'checkpoint_model': checkpoint_model,
            'early_stopping_patience': early_stopping_patience,
            'batch_size': batch_size,
            'trained': False,
            'evaluate_model': evaluate_model,
            'debug': debug,
            'custom_tokenizer': None,
            'train_data_csv': train_data_csv,
            'aug_data_csv': aug_data_csv,
            'test_size': test_size,
            'preload_train_data_dill': preload_train_data_dill,
            'save_train_data_dill': save_train_data_dill,
            'features_to_train': None,
            'custom_text_input_length': 50,
            'load_to_predict': False,
            'model_h5': model_h5
        }

        if os.path.exists(json_settings):
            with open(json_settings, 'r') as f:
                data = json.load(f)

            # Replace arg dict values with json values
            for key in data.keys():
                if key in settings_dict.keys():
                    settings_dict[key] = data[key]

        return settings_dict


class TwitterSentimentModelInterface(TwitterModelInterface):

    @staticmethod
    def process_sentiment_model_args(**kwargs) -> dict:
        return TwitterModelInterface.get_settings_dict(**kwargs)

    @staticmethod
    def create_sentiment_model_to_train(**kwargs) -> TwitterSentimentModel.SentimentModelLearning:
        """
        Creates instances of SentimentModelData, SentimentModelParameters,
        and SentimentModelLearning in a way that these objects are prepared for training the model.

        :param kwargs: Any keyword arguments which are described in process_sentiment_model_args
        :type kwargs: str: any

        :return: Initialized and trained Twitter sentiment model
        :rtype: SentimentModelLearning
        """

        settings_dict = TwitterSentimentModelInterface.process_sentiment_model_args(**kwargs)

        parameters = ModelBase.ModelParameters(**settings_dict)

        data = TwitterSentimentModel.SentimentModelData(parameters)

        model = TwitterSentimentModel.SentimentModelLearning(parameters, data)
        model.build_model()

        return model


class TwitterSpamModelInterface(TwitterModelInterface):

    @staticmethod
    def process_spam_model_args(**kwargs) -> dict:

        if not kwargs['features_to_train']:
            kwargs['features_to_train'] = ['full_text', 'cap.english', 'cap.universal',
                                           'raw_scores.english.overall',
                                           'raw_scores.universal.overall',
                                           'raw_scores.english.astroturf',
                                           'raw_scores.english.fake_follower',
                                           'raw_scores.english.financial',
                                           'raw_scores.english.other',
                                           'raw_scores.english.self_declared',
                                           'raw_scores.english.spammer',
                                           'raw_scores.universal.astroturf',
                                           'raw_scores.universal.fake_follower',
                                           'raw_scores.universal.financial',
                                           'raw_scores.universal.other',
                                           'raw_scores.universal.self_declared',
                                           'raw_scores.universal.spammer',
                                           'favorite_count', 'retweet_count']

        return TwitterModelInterface.get_settings_dict(**kwargs)

    @staticmethod
    def classify_twitter_query(keywords, num: int, dill_file: str, filename=None,
                               use_botometer_lite=False) -> pd.DataFrame:
        """
        Queries tweets and their botscores, loads spam model from dill and h5, predicts on tweets, and returns tweet
        dataframe with spam model predictions.

        :param keywords: Tweet keywords to query
        :type keywords: list(str)
        :param num: Number of Tweets to query per keyword
        :type num: int
        :param dill_file: Path to dill file which stores SpamModelParameters
        :type dill_file: str
        :param filename: Filename to save labeled dataframe to
        :type filename: str
        :param use_botometer_lite: Whether or not to use botometer lite (when querying Tweets)
        :type use_botometer_lite: bool

        :return: Dataframe of Tweets with Spam Model labels
        :rtype: pd.DataFrame
        """
        if filename is None:
            filename = '&'.join(keywords)
            filename += f'x{num}'
            filename += '.csv'

        # Tweet query
        tdm = TweetDatabaseManager(use_botometer_lite=use_botometer_lite)
        df = tdm.save_multiple_keywords(keywords, num, same_file=True, filename=None, save_to_file=False)

        spam_model_learning = TwitterSpamModelInterface.load_spam_model_to_predict(dill_file)

        # Parse Tweet df to make sure it has needed keys
        df = Utils.parse_json_tweet_data(df, spam_model_learning.parameters.features_to_train)

        df['SpamModelLabel'] = spam_model_learning.predict(tweet_df=df)

        Utils.write_dataframe_to_csv(df, filename, write_index=False)

        return df

    @staticmethod
    def create_spam_model_to_train(**kwargs) -> TwitterSpamModel.SpamModelLearning:
        """
        Creates instances of SpamModelData, SpamModelParameters, and SpamModelLearning in a way that these
        objects are prepared for training the model.

        :param kwargs: Any keyword arguments which are described in process_spam_model_args
        :type kwargs: str: any

        :return: Initialized and trained Twitter spam model
        :rtype: SpamModelLearning
        """

        settings_dict = TwitterSpamModelInterface.process_spam_model_args(**kwargs)

        parameters = ModelBase.ModelParameters(**settings_dict)

        data = TwitterSpamModel.SpamModelData(parameters)

        model = TwitterSpamModel.SpamModelLearning(parameters, data)
        model.build_model()

        return model

    @staticmethod
    def preprocess_training_data(json_settings='../data/Learning Data/spam_settings.json',
                                 train_data_csv='../data/Learning Data/spam_train.csv',
                                 aug_data_csv='../data/Learning Data/spam_train_aug.csv',
                                 test_size=0.1, save_train_data_dill='../data/Learning Data/preload.dill',
                                 features_to_train=None) -> bool:
        """
        Loads csv training data and processes it (sanitizes, vectorizes, molds into model input shape). Saves this to a
        dill file for use whenever loading the model to train. Parameters passed here are the only ones needed for data
        to be preprocessed. If parameters here need to be changed, then data needs to be reprocessed before training.

        See process_spam_model_args() for descriptions of args

        :return: Whether or not the preprocessing was successful and the save_train_data_dill file exists
        :rtype: bool
        """

        successful = False
        settings = TwitterSpamModelInterface.process_spam_model_args(json_settings=json_settings,
                                                                     train_data_csv=train_data_csv,
                                                                     aug_data_csv=aug_data_csv,
                                                                     test_size=test_size,
                                                                     save_train_data_dill=save_train_data_dill,
                                                                     features_to_train=features_to_train)

        parameters = ModelBase.ModelParameters(**settings)
        _ = TwitterSpamModel.SpamModelData(parameters)

        if os.path.isfile(save_train_data_dill):
            successful = True

        return successful

    @staticmethod
    def save_trained_model(sml: TwitterSpamModel.SpamModelLearning, dill_parameters_file: str,
                           h5_file: str = '') -> bool:
        """
        Saves a trained SpamModelLearning to an h5 file and its SpamModelParameters class to a dill file in a format
        that the model can be loaded in prediction mode.

        :param sml: Trained SpamModelLearning
        :type sml: SpamModelLearning
        :param dill_parameters_file: Path to the dill file in which SpamModelParameters will be saved.
        :type: str
        :param h5_file: Optional path to the h5 file where SpamModelLearning model will be saved (otherwise uses the h5
                        path in SpamModelLearning.SpamModelParameters.h5)
        :type h5_file: str

        :return: Whether the saving operation was successful
        :rtype: bool
        """

        successful = False

        if h5_file:
            sml.parameters.h5 = h5_file

        # Save h5 of model
        sml.model.save(sml.parameters.h5)

        # Make sure parameters class is updated before saving
        sml.update_to_save_as_trained(sml.data.nsc, sml.data.text_input_length)

        # Save parameters to dill
        with open(dill_parameters_file, 'wb') as dpf:
            dill.dump(sml.parameters, dpf)

        # Check that files exist
        if os.path.isfile(dill_parameters_file) and os.path.isfile(sml.parameters.h5):
            successful = True

        return successful

    @staticmethod
    def load_spam_model_to_predict(dill_parameters_file: str) -> TwitterSpamModel.SpamModelLearning:
        """
        Loads an instance of SpamModelData, SpamModelParameters, and SpamModelLearning from saved model .h5 and
        saved parameters .dill files in a way that these objects are prepared for using the model to predict.

        :param dill_parameters_file: Path to dill file which stores SpamModelParameters
        :type dill_parameters_file: str

        :return: A compiled SpamModelLearning ready to make predictions
        :rtype: SpamModelLearning
        """
        with open(dill_parameters_file, 'rb') as dpf:
            parameters = dill.load(dpf)

        data = TwitterSpamModel.SpamModelData(parameters)
        model = TwitterSpamModel.SpamModelLearning(parameters, data)
        model.build_model()

        return model
