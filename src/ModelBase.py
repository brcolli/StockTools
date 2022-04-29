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
                 early_stopping=False,
                 checkpoint_model=False,
                 early_stopping_patience=0,
                 batch_size=128,
                 trained=False,
                 evaluate_model=True,
                 debug=False,
                 score=(-1, -1),

                 custom_tokenizer=None,
                 train_data_csv='',
                 aug_data_csv='',
                 test_size=0.1,
                 preload_train_data_dill='',
                 save_train_data_dill='',
                 features_to_train=None,
                 custom_text_input_length=44,

                 load_predict_only=False,
                 model_h5=''
                 ):

        # Learning Related Parameters
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.checkpoint_model = checkpoint_model
        self.early_stopping_patience = early_stopping_patience
        self.batch_size = batch_size
        self.trained = trained
        self.evaluate_model_on_load = evaluate_model
        self.debug = debug
        self.score = score

        # Data Related Parameters
        self.custom_tokenizer = custom_tokenizer
        self.train_data_csv = train_data_csv
        self.aug_data_csv = aug_data_csv
        self.test_size = test_size
        self.preload_train_data_dill = preload_train_data_dill
        self.save_train_data_dill = save_train_data_dill
        self.features_to_train = features_to_train
        self.textless_features_to_train = []
        self.custom_text_input_length = custom_text_input_length

        # Performance Related Parameters
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f_score = 0
        self.mcor = 0
        self.f_beta = 0

        # Loading and Mode
        self.load_to_train = not load_predict_only
        self.load_to_predict = load_predict_only
        self.h5 = model_h5

    def update_to_save_as_trained(self, nsc, text_input_length):
        self.trained = True
        self.custom_tokenizer = nsc.tokenizer
        self.custom_text_input_length = text_input_length
        self.train_data_csv = ''
        self.aug_data_csv = ''
        self.preload_train_data_dill = ''
        self.save_train_data_dill = ''
        self.load_to_train = False
        self.load_to_predict = True


class ModelData:
    """This single class should represent all the data being used in the model. Point is that functions to load and
    process data are contained within the class. In case new data is passed, the class processes it. Otherwise, the
    class should load itself from a pickle.
    """

    def __init__(self, parameters: ModelParameters):
        self.parameters = parameters

        self.nsc = NSC()
        if self.parameters.custom_tokenizer:
            self.nsc.tokenizer = self.parameters.custom_tokenizer

        self.text_input_length = self.parameters.custom_text_input_length

        # ModelData specific parameters
        self.x_train_text_embeddings = None
        self.x_test_text_embeddings = None
        self.x_train_meta = None
        self.x_test_meta = None
        self.glove_embedding_matrix = None
        self.y_train = None
        self.y_test = None

    # def load_data_from_feather(self):
    #     feather_file = self.parameters.preload_train_data_feather
    #     # Need to load:
    #     # self.x_train_text_embeddings, self.x_test_text_embeddings, self.x_train_meta, self.x_test_meta, \
    #     # self.glove_embedding_matrix, self.y_train, self.y_test
    #     # from feather_file
    #
    # def save_data_to_feather(self):
    #     to_feather_file = self.parameters.save_train_data_feather
    #     # Need to save:
    #     # self.x_train_text_embeddings, self.x_test_text_embeddings, self.x_train_meta, self.x_test_meta, \
    #     # self.glove_embedding_matrix, self.y_train, self.y_test
    #     # to to_feather_file

    def load_data_from_dill(self):
        """
        Loads model data from a binary file
        :param from_preload_binary: Path to preload binary
        :type from_preload_binary: str
        """

        if os.path.exists(self.parameters.preload_train_data_dill):
            with open(self.parameters.preload_train_data_dill, "rb") as fpb:
                data = pickle.load(fpb)

            self.x_train_text_embeddings, self.x_test_text_embeddings, self.x_train_meta, self.x_test_meta, \
            self.glove_embedding_matrix, self.y_train, self.y_test, self.nsc.tokenizer, self.text_input_length = data

    def save_data_to_dill(self):
        """
        Saves data to a preload binary (so it can then be loaded using load_data_from_binary)
        :param save_preload_binary: Path to preload binary
        :type save_preload_binary: str
        """
        with open(self.parameters.save_train_data_dill, 'wb') as f:
            data = (self.x_train_text_embeddings, self.x_test_text_embeddings, self.x_train_meta, self.x_test_meta,
                    self.glove_embedding_matrix, self.y_train, self.y_test, self.nsc.tokenizer, self.text_input_length)
            pickle.dump(data, f)


class ModelLearning:
    """This is the main Model class. It will create objects from Data() and Version() classes. Added functionality will be
    to load the entire model in from a file, get version info on previous models, reuse data classes in case the same data
    is to be used with different hyperparamaters, and to add model testing / operation functionality.
    """

    def __init__(self):

        self.model = tf.keras.models.Model