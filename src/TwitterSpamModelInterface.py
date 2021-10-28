import dill
import json
import operator

# Used for convenience as this module is pretty much a wrapper for TwitterSpamModel.py
from TwitterSpamModel import *

"""TwitterSpamModelInterface

Description:
Module to work with the TwitterSpamModel.py class with functions to load the model from origin, load the model from bin,
or save the model to bin. This module needs to be separate from TwitterSpamModel.py so that the dill library
(saving library) works properly. 

Authors: Fedya Semenov
Date: October 13, 2021
"""


def load_model_from_bin(path):
    """
    Loads and returns a model object from a binary save file

    :param path: Path to the saved model
    :type path: str

    :return: Loaded model
    :rtype: SpamModelLearning
    """
    with open(path, 'rb') as mb:
        data = dill.load(mb)

        # If model has been trained, we must load the model.model using tensorflow method
        if data[0]:
            _, model_path, model_data, params = data
            model = SpamModelLearning(params, model_data)  # Params and model_data stored normally in pickle
            model.model = tf.keras.models.load_model(model_path)  # Model stored through tensorflow
            model.parameters.trained = True

        # If model not trained yet, load normally from pickle
        else:
            _, model = data
    return model


def save_model_to_bin(path, model):
    """
    Saves a model object to a binary save file

    :param path: Path to save the model at
    :type path: str
    :param model: Twitter spam model instance to be saved
    :type model: SpamModelLearning
    """
    bin_path = path + '.bin'

    # If model has been trained, save using tensorflow method and pickle
    if model.parameters.trained:
        data = model.data
        params = model.parameters
        model_path = path + '.tf'
        model.model.save(model_path)
        bin_data = (True, model_path, data, params)

    # If model not trained yet, save normally to pickle
    else:
        bin_data = (False, model)

    with open(bin_path, 'wb') as mb:
        dill.dump(bin_data, mb)


def load_model_from_origin(base_data_csv='', test_size=0.3, features_to_train=None, aug_data_csv=None,
                           save_preload_data_to_bin='', from_preload_data_bin='', epochs=100, learning_rate=0.0001,
                           saved_model_bin='', early_stopping=False, load_model=False, early_stopping_patience=0,
                           batch_size=128, settings_file=''):
    """
    Loads and returns a model object from provided data with provided settings

    :param base_data_csv: Path of the base data csv file (containing Tweet dataframe in csv format)
    :type base_data_csv: str
    :param test_size: The test size (or portion of data to be used for test set during training)
    :type test_size: float 0 < x < 1
    :param features_to_train: Tweet and botometer features to train the model on
    :type features_to_train: list(str)
    :param aug_data_csv: Path of the augmented data csv file (containing augmented Tweet dataframe in csv format)
    :param save_preload_data_to_bin: Path to save model data into a preload bin (this saves time in case you are
                                    initializing another model with the same data)
    :type save_preload_data_to_bin: str
    :param from_preload_data_bin: Path to preloaded model data saved in a bin (saves time in case you saved your preload
                                    data on previous model initialization)
    :type from_preload_data_bin: str
    :param epochs: Number of epochs to train the model on, can be changed later, before training model
    :type epochs: int
    :param saved_model_bin: Path to save the model at for early stopping (not the same as saving the entire Model Class
                            (SpamModelLearning) through save_model_to_bin)
    :type saved_model_bin: str
    :param early_stopping: Whether or not to use Early Stopping during model training
    :type early_stopping: bool
    :param load_model: Whether or not to load the model (when starting training) from a previously saved callback (not
                        the same as loading the entire Model Class (SpamModelLearning) through load_model_from_bin)
    :type load_model: str
    :param early_stopping_patience: In the case of early stopping, number of epochs to run without improvement before
                                    stopping the training
    :type early_stopping_patience: int
    :param batch_size: Batch size to train the model with
    :type batch_size: int
    :param settings_file: Path to an optional json file to load all of the aforementioned parameters from
    :type settings_file: str

    :return: Initialized, but not built (not trained) Twitter model
    :rtype: SpamModelLearning
    """
    if os.path.exists(settings_file):
        with open(settings_file, 'r') as f:
            data = json.load(f)

        base_data_csv, test_size, features_to_train, aug_data_csv, save_preload_data_to_bin, from_preload_data_bin, \
        epochs, learning_rate, saved_model_bin, early_stopping, load_model, early_stopping_patience, batch_size = \
            operator.itemgetter('base_data_csv', 'test_size', 'features_to_train', 'aug_data_csv',
                                'save_preload_data_to_bin', 'from_preload_data_bin', 'epochs', 'learning_rate',
                                'saved_model_bin', 'early_stopping', 'load_model', 'early_stopping_patience',
                                'batch_size')(data)

    if features_to_train is None:
        features_to_train = ['full_text']

    nsc = NSC()
    # parameters = SpamModelParameters(epochs, saved_model_bin, early_stopping, load_model, early_stopping_patience,
    #                                  batch_size, False, False)

    parameters = SpamModelParameters(learning_rate=learning_rate,
                                     epochs=epochs,
                                     saved_model_bin=saved_model_bin,
                                     early_stopping=early_stopping,
                                     checkpoint_model=False,
                                     load_model=load_model,
                                     early_stopping_patience=early_stopping_patience,
                                     batch_size=batch_size,
                                     trained=False,
                                     debug=False)

    data = SpamModelData(nsc, base_data_csv, test_size, features_to_train, aug_data_csv=aug_data_csv,
                         save_preload_binary=save_preload_data_to_bin, from_preload_binary=from_preload_data_bin)
    model = SpamModelLearning(parameters, data)

    return model
