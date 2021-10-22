import os
import pickle
import tensorflow as tf
import NLPSentimentCalculations

NSC = NLPSentimentCalculations.NLPSentimentCalculations


class ModelParameters:
    """Idea of this class is to be called to check on the version of the model and get info about the model's training.
    """

    def __init__(self,
                 learning_rate=1e-3,
                 epochs=100,
                 saved_model_bin='',
                 early_stopping=False,
                 checkpoint_model=False,
                 load_model=False,
                 early_stopping_patience=0,
                 batch_size=128,
                 trained=False,
                 debug=False):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.saved_model_bin = saved_model_bin
        self.early_stopping = early_stopping
        self.checkpoint_model = checkpoint_model
        self.load_model = load_model
        self.early_stopping_patience = early_stopping_patience
        self.batch_size = batch_size
        self.trained = trained
        self.debug = debug
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f_score = 0


class ModelData:
    """This single class should represent all the data being used in the model. Point is that functions to load and process
    data are contained within the class. In case new data is passed, the class processes it. Otherwise, the class should
    load itself from a pickle.
    """

    def __init__(self, nsc, base_data_csv, test_size, features_to_train, aug_data_csv=None):

        self.nsc = nsc
        self.base_data_csv = base_data_csv
        self.test_size = test_size
        self.text_input_length = 0

        self.features_to_train = features_to_train
        if features_to_train is None:
            self.features_to_train = ['full_text']

        self.aug_data_csv = aug_data_csv

        self.x_train_text_embeddings = None
        self.x_test_text_embeddings = None
        self.x_train_meta = None
        self.x_test_meta = None
        self.glove_embedding_matrix = None
        self.y_train = None
        self.y_test = None

    def load_data_from_binary(self, from_preload_binary):
        """
        Loads model data from a binary file
        :param from_preload_binary: Path to preload binary
        :type from_preload_binary: str
        """

        if os.path.exists(from_preload_binary):
            with open(from_preload_binary, "rb") as fpb:
                data = pickle.load(fpb)

            self.x_train_text_embeddings, self.x_test_text_embeddings, self.x_train_meta, self.x_test_meta, \
            self.glove_embedding_matrix, self.y_train, self.y_test, self.nsc.tokenizer = data

    def save_data_to_binary(self, save_preload_binary):
        """
        Saves data to a preload binary (so it can then be loaded using load_data_from_binary)
        :param save_preload_binary: Path to preload binary
        :type save_preload_binary: str
        """
        with open(save_preload_binary, 'w') as f:
            data = (self.x_train_text_embeddings, self.x_test_text_embeddings, self.x_train_meta, self.x_test_meta,
                    self.glove_embedding_matrix, self.y_train, self.y_test, self.nsc.tokenizer)
            pickle.dump(data, f)


class ModelLearning:
    """This is the main Model class. It will create objects from Data() and Version() classes. Added functionality will be
    to load the entire model in from a file, get version info on previous models, reuse data classes in case the same data
    is to be used with different hyperparamaters, and to add model testing / operation functionality.
    """

    def __init__(self):

        self.model = tf.keras.models.Model
