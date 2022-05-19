import operator
import json

from TwitterSentimentModel import *


class TwitterSentimentModelInterface:

    @staticmethod
    def create_sentiment_model(nsc=None,
                               test_size=0.1,
                               learning_rate=1e-3,
                               epochs=1000,
                               early_stopping=False,
                               checkpoint_model=False,
                               load_model=False,
                               early_stopping_patience=0,
                               batch_size=128,
                               trained=False,
                               evaluate_model=True,
                               debug=False,
                               aug_data_csv=None,
                               save_preload_binary='',
                               from_preload_binary='',
                               test_model=False,
                               base_data_csv='../data/Learning Data/sentiment_learning.csv',
                               test_set_csv='../data/Learning Data/sentiment_test_set.csv',
                               saved_model_bin='../data/analysis/Model Results/Saved Models/best_sentiment_model.h5'):

        sent_model_params = ModelParameters(learning_rate=learning_rate,
                                            epochs=epochs,
                                            trained=trained,
                                            early_stopping=early_stopping,
                                            early_stopping_patience=early_stopping_patience,
                                            batch_size=batch_size,
                                            load_model=load_model,
                                            checkpoint_model=checkpoint_model,
                                            saved_model_bin=saved_model_bin,
                                            evaluate_model=evaluate_model,
                                            debug=debug)

        if not nsc:
            nsc = NSC()

        sent_model_data = SentimentModelData(nsc=nsc, base_data_csv=base_data_csv,
                                             test_size=test_size,
                                             aug_data_csv=aug_data_csv, save_preload_binary=save_preload_binary,
                                             from_preload_binary=from_preload_binary)

        sent_model_learning = SentimentModelLearning(sent_model_params, sent_model_data)
        sent_model_learning.build_model()

        if test_model:
            sent_score_dict = sent_model_learning.predict_and_score(test_set_csv)

            print(sent_score_dict)

        return sent_model_learning

    @staticmethod
    def load_model_from_origin(base_data_csv='', test_size=0.3, aug_data_csv=None,
                               save_preload_data_to_bin='', from_preload_data_bin='', epochs=100, learning_rate=0.0001,
                               saved_model_bin='', early_stopping=False, load_model=False, early_stopping_patience=0,
                               batch_size=128, settings_file=''):
        """
        Loads and returns a model object from provided data with provided settings

        :param base_data_csv: Path of the base data csv file (containing Tweet dataframe in csv format)
        :type base_data_csv: str
        :param test_size: The test size (or portion of data to be used for test set during training)
        :type test_size: float 0 < x < 1
        :param aug_data_csv: Path of the augmented data csv file (containing augmented Tweet dataframe in csv format)
        :param save_preload_data_to_bin: Path to save model data into a preload bin (this saves time in case you are
                                        initializing another model with the same data)
        :type save_preload_data_to_bin: str
        :param from_preload_data_bin: Path to preloaded model data saved in a bin (saves time in case you saved your
                                      preload data on previous model initialization)
        :type from_preload_data_bin: str
        :param epochs: Number of epochs to train the model on, can be changed later, before training model
        :type epochs: int
        :param learning_rate: Learning rate of model
        :type learning_rate: float
        :param saved_model_bin: Path to save the model at for early stopping (not the same as saving the entire Model
                                Class (SentimentModelLearning) through save_model_to_bin)
        :type saved_model_bin: str
        :param early_stopping: Whether or not to use Early Stopping during model training
        :type early_stopping: bool
        :param load_model: Whether or not to load the model (when starting training) from a previously saved callback
                           (not the same as loading the entire Model Class (SentimentModelLearning) through
                           load_model_from_bin)
        :type load_model: str
        :param early_stopping_patience: In the case of early stopping, number of epochs to run without improvement
                                        before stopping the training
        :type early_stopping_patience: int
        :param batch_size: Batch size to train the model with
        :type batch_size: int
        :param settings_file: Path to an optional json file to load all of the aforementioned parameters from
        :type settings_file: str

        :return: Initialized, but not built (not trained) Twitter model
        :rtype: SentimentModelLearning
        """

        if os.path.exists(settings_file):
            with open(settings_file, 'r') as f:
                data = json.load(f)

            base_data_csv, test_size, features_to_train, aug_data_csv, save_preload_data_to_bin, \
            from_preload_data_bin, epochs, learning_rate, saved_model_bin, early_stopping, load_model, \
            early_stopping_patience, batch_size = \
                operator.itemgetter('base_data_csv', 'test_size', 'features_to_train', 'aug_data_csv',
                                    'save_preload_data_to_bin', 'from_preload_data_bin', 'epochs', 'learning_rate',
                                    'saved_model_bin', 'early_stopping', 'load_model', 'early_stopping_patience',
                                    'batch_size')(data)

        nsc = NSC()

        parameters = ModelParameters(learning_rate=learning_rate,
                                     epochs=epochs,
                                     saved_model_bin=saved_model_bin,
                                     early_stopping=early_stopping,
                                     checkpoint_model=False,
                                     load_model=load_model,
                                     early_stopping_patience=early_stopping_patience,
                                     batch_size=batch_size,
                                     trained=False,
                                     debug=False)

        data = SentimentModelData(nsc, base_data_csv, test_size, aug_data_csv=aug_data_csv,
                                  save_preload_binary=save_preload_data_to_bin,
                                  from_preload_binary=from_preload_data_bin)
        model = SentimentModelLearning(parameters, data)

        return model
