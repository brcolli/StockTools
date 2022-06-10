import os
from abc import ABC, abstractmethod
from typing import List, Tuple
import pickle
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from NLPSentimentCalculations import NLPSentimentCalculations as nSC
from dataclasses import dataclass
from utilities import Utils


Metrics = ['acc', nSC.precision, nSC.recall, nSC.mcor,
           tfa.metrics.FBetaScore(num_classes=2, average='weighted', beta=1.0, name='fbeta')]
MetricsKeys = ['acc', 'precision', 'recall', 'mcor', 'fbeta']
MetricsDict = dict(zip(MetricsKeys, Metrics))

"""Idea of this module is to be the parent class for the Twitter models, containing all shared methods and attributes
such as hyper parameters and model initialization.
"""


@dataclass
class ModelParameters:

    # Learning Related Parameters
    learning_rate: float = 1E-3
    epochs: int = 1000
    early_stopping: bool = False
    checkpoint_model: bool = False
    early_stopping_patience: int = 0
    batch_size: int = 128
    evaluate_model: bool = True
    debug: bool = False
    use_tpu: bool = False
    use_transformers: bool = True

    # Data Related Parameters
    custom_tokenizer: object = None
    train_data_csv: str = ''
    aug_data_csv: str = ''
    test_size: float = 0.1
    preload_train_data_dill: str = ''
    save_train_data_dill: str = ''
    features_to_train: list = None
    textless_features_to_train: list = None
    custom_text_input_length: int = 50

    # Performance Related Parameters
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f_score: float = 0.0
    mcor: float = 0.0

    # Loading and Mode
    load_to_predict: bool = False
    model_h5: str = ''


class ModelData(ABC):

    """This single class should represent all the data being used in the model. Point is that functions to load and
    process data are contained within the class. In case new data is passed, the class processes it. Otherwise, the
    class should load itself from a pickle.
    """

    def __init__(self, parameters: ModelParameters):
        self.parameters = parameters

        self.nsc = nSC()
        if self.parameters.custom_tokenizer:
            self.nsc.tokenizer = self.parameters.custom_tokenizer

        self.text_input_length = self.parameters.custom_text_input_length

        # ModelData specific parameters
        self.train_text_input_ids = None
        self.test_text_input_ids = None
        self.x_train_meta = None
        self.x_test_meta = None
        self.train_embedding_mask = None
        self.test_embedding_mask = None
        self.y_train = None
        self.y_test = None

    def get_x_val_from_csv(self, csv: str):
        """
        Loads an x_validation dataset from a csv, in a format ready to pass into model.predict

        :param csv: Path to csv file containing a saved dataframe of tweets with self.features_to_train columns present
        :type csv: str

        :return: Data ready to be passed into the model for prediction
        :rtype: x_val_text_embeddings
        """

        df = Utils.parse_json_tweet_data_from_csv(csv, self.parameters.features_to_train)
        return self.get_x_val_from_dataframe(df)

    @abstractmethod
    def get_x_val_from_dataframe(self, x_val: pd.DataFrame):
        pass

    def get_vectorized_text_tokens_from_val_dataframe(self, x_val: pd.DataFrame) -> Tuple[List[str], List[str]]:

        """
        Generates vectorized text tokens from a dataframe that has been sorted for validation.

        :param x_val: Dataframe of input values to vectorize.
        :type x_val: pd.Dataframe

        :return: List of vectorized input
        :rtype: List[str]
        """

        if 'full_text' in self.parameters.features_to_train:
            x_val_text_data = x_val['full_text']
        else:
            x_val_text_data = pd.DataFrame()

        # Sanitizes textual data
        x_val_text_clean = [nSC.sanitize_text_string(s) for s in list(x_val_text_data)]

        # Vectorizes textual data
        if self.parameters.use_transformers:

            x_val_text_embeddings = self.nsc.tokenizer(x_val_text_clean, padding=True, truncation=True,
                                                       return_tensors='tf')

            val_text_input_ids = x_val_text_embeddings['input_ids']
            val_embedding_mask = x_val_text_embeddings['attention_mask']

        else:
            _, val_text_input_ids = self.nsc.keras_word_embeddings(x_val_text_clean, self.text_input_length)
            val_embedding_mask = self.train_embedding_mask

        return val_text_input_ids, val_embedding_mask

    def get_vectorized_text_tokens_from_dataframes(self, x_train_text_data: pd.DataFrame,
                                                   x_test_text_data: pd.DataFrame) -> Tuple[List[str],
                                                                                            List[str],
                                                                                            List[str],
                                                                                            List[str]]:
        """
        Generates vectorized text tokens from train and test dataframes.

        :param x_train_text_data: Dataframe of train input values to vectorize.
        :type x_train_text_data: pd.Dataframe
        :param x_test_text_data: Dataframe of test input values to vectorize.
        :type x_test_text_data: pd.Dataframe

        :return: List of vectorized input
        :rtype: List[str]
        """

        # Clean the textual data
        x_train_text_clean = [nSC.sanitize_text_string(s) for s in list(x_train_text_data)]
        x_test_text_clean = [nSC.sanitize_text_string(s) for s in list(x_test_text_data)]

        # Initialize tokenizer on training data
        if self.parameters.use_transformers:

            self.nsc.create_roberta_tokenizer()

            x_train_text_embeddings = self.nsc.tokenizer(x_train_text_clean, padding=True, truncation=True,
                                                         return_tensors='tf')
            x_test_text_embeddings = self.nsc.tokenizer(x_test_text_clean, padding=True, truncation=True,
                                                        return_tensors='tf')

            train_text_input_ids = x_train_text_embeddings['input_ids']
            train_embedding_mask = x_train_text_embeddings['attention_mask']

            test_text_input_ids = x_test_text_embeddings['input_ids']
            test_embedding_mask = x_test_text_embeddings['attention_mask']

            self.text_input_length = -1

        else:
            self.nsc.tokenizer.fit_on_texts(x_train_text_clean)

            # Create word vectors from tokens
            self.text_input_length, train_text_input_ids = self.nsc.keras_word_embeddings(x_train_text_clean)

            _, test_text_input_ids = self.nsc.keras_word_embeddings(x_test_text_clean, self.text_input_length)

            train_embedding_mask = self.nsc.create_glove_word_vectors()
            test_embedding_mask = train_embedding_mask

        return train_text_input_ids, test_text_input_ids, train_embedding_mask, test_embedding_mask

    def get_twitter_dataframe_from_csv(self) -> pd.DataFrame:

        """
        Creates a dataframe from a CSV of tweets
        """

        twitter_df = Utils.parse_json_tweet_data_from_csv(self.parameters.train_data_csv,
                                                          self.parameters.features_to_train)

        if 'augmented' not in twitter_df.columns:
            twitter_df['augmented'] = 0

        if self.parameters.aug_data_csv:
            aug_df = Utils.parse_json_tweet_data_from_csv(self.parameters.aug_data_csv,
                                                          self.parameters.features_to_train)
            if 'augmented' not in aug_df.columns:
                aug_df['augmented'] = 1

            twitter_df = pd.concat([twitter_df, aug_df])

        return twitter_df

    def load_data_from_dill(self):
        """
        Loads model data from a binary file
        """

        if os.path.exists(self.parameters.preload_train_data_dill):

            with open(self.parameters.preload_train_data_dill, "rb") as fpb:
                data = pickle.load(fpb)

            self.train_text_input_ids, self.test_text_input_ids, self.x_train_meta, self.x_test_meta, \
                self.train_embedding_mask, self.test_embedding_mask, self.y_train, self.y_test, self.nsc.tokenizer,\
                self.text_input_length = data

    def save_data_to_dill(self):
        """
        Saves data to a preload binary (so it can then be loaded using _load_data_from_binary)
        """
        with open(self.parameters.save_train_data_dill, 'wb') as f:
            data = (self.train_text_input_ids, self.test_text_input_ids, self.x_train_meta, self.x_test_meta,
                    self.train_embedding_mask, self.test_embedding_mask, self.y_train, self.y_test, self.nsc.tokenizer,
                    self.text_input_length)
            pickle.dump(data, f)

    @abstractmethod
    def get_dataset_from_tweet_type(self, dataframe: pd.DataFrame):
        pass

    @abstractmethod
    def load_data_from_csv(self):
        pass


class ModelLearning:
    """This is the main Model class. It will create objects from Data() and Version() classes. Added functionality will
    be to load the entire model in from a file, get version info on previous models, reuse data classes in case the same
    data is to be used with different hyperparamaters, and to add model testing / operation functionality.
    """

    def __init__(self, model_params: ModelParameters, model_data: ModelData):

        self.parameters = model_params
        self.data = model_data
        self.metrics = Metrics
        self.model = tf.keras.models.Model
        self.tpu_strategy = None
        self.score = (-1, -1)

    def compile_model(self):

        """
        Compiles a model with given hyperparameters
        """

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.parameters.learning_rate, clipnorm=1.)
        self.model.compile(loss='binary_crossentropy',
                           optimizer=optimizer,
                           metrics=self.metrics)
        print(self.model.summary())  # Print model summary

    def raw_predict_from_x_val(self, x_val: pd.DataFrame):
        """
        Predicts on a dataframe of validation vectorized data.

        :param x_val: Dataframe of vectorized data.
        :type x_val: pd.Dataframe

        :return: Softmax probabilities for each label (-1, 0, and 1) of each data
        :rtype: [[x, y, z]] where x, y, z are floats in range (0, 1) and x + y + z = 1.00
        """
        return self.model.predict(x_val)

    def evaluate_model(self, test_input_layer, test_labels, cbs) -> Tuple[float, float]:
        """
        Evaluate model using tensorflow methods

        :param test_input_layer: Model input layer being [x_val_text_embeddings, x_val_meta] or x_val_text_embeddings
        :type test_input_layer: list
        :param test_labels: Tweet labels
        :type test_labels: list(int)
        :param cbs: List of callbacks
        :type cbs: list(func)

        :return: score and accuracy tuple
        :rtype: (float, float)
        """

        score = self.model.evaluate(x=test_input_layer, y=test_labels, verbose=1, callbacks=cbs)

        print("Test Score:", score[0])
        print("Test Accuracy:", score[1])

        return score

    def raw_predict_csv(self, csv: str):
        """
        Returns raw model prediction scores on tweets from a csv. CSV must be a saved dataframe of tweets with
        all the columns in self.data.features_to_train present. Raw model prediction scores are currently formatted as
        softmax probabilities of Tweet labels: [[Probability of -1, P(0), P(1)], ...] where P(-1) + P(0) + P(1) = 1 for
        any given tweet. Format: [[Tweet 1 P(-1), Tweet 1 P(0), Tweet 1 P(1)], [T2 P(-1), T2 P(0), T2 P(1)], ...].

        :param csv: Filepath to dataframe of tweets with self.data.features_to_train columns
        :type csv: str

        :return: Softmax probabilities for each label (-1, 0, and 1) of each tweet
        :rtype: [[x, y, z]] where x, y, z are floats in range (0, 1) and x + y + z = 1.00
        """
        x_val = self.data.get_x_val_from_csv(csv)
        return self.model.predict(x_val).tolist()

    def raw_predict_tweets(self, tweet_df: pd.DataFrame):
        """
        Predict on model from a dataframe of tweets.

        :param tweet_df: Dataframe of tweets.
        :type tweet_df: pd.Dataframe

        :return: Softmax probabilities for each label (-1, 0, and 1) of each tweet
        :rtype: [[x, y, z]] where x, y, z are floats in range (0, 1) and x + y + z = 1.00
        """
        x_val = self.data.get_x_val_from_dataframe(tweet_df)
        return self.model.predict(x_val)

    def predict(self, csv: str = '', tweet_df: pd.DataFrame = None):
        """
        Predicts Tweet labels from a csv of Tweets. CSV must be a saved dataframe of tweets with all the
        columns in self.data.features_to_train present. Each prediction is either a -1 (not marked), 0 (clean), or
        1 (spam).

        :param csv: Filepath to dataframe of tweets with self.data.features_to_train columns
        :type csv: str
        :param tweet_df: Dataframe of tweet data
        :type tweet_df: pandas.core.frame.DataFrame

        :return: Prediction for each tweet
        :rtype: [int] where int is -1, 0, or 1
        """

        # Get the raw softmax probability predictions
        if csv:
            y = self.raw_predict_csv(csv)
        elif not tweet_df.empty:
            y = self.raw_predict_tweets(tweet_df)
        else:
            return []

        # Use the highest softmax probability as the label (-1, 0, or 1)
        return [max(range(len(y1)), key=y1.__getitem__) for y1 in y]

    def predict_and_score(self, csv, affect_parameter_scores=False):
        """
        Predicts Tweet labels from a csv of Tweets. Then scores those predictions using labels provided in the csv. CSV
        must be a saved dataframe of tweets with all the columns in self.data.features_to_train + 'Label' present.

        :param csv: Filepath to dataframe of tweets with self.data.features_to_train and 'Label' columns
        :type csv: str
        :param affect_parameter_scores: Whether or not to write the generated scores to self.parameters
        :type affect_parameter_scores: bool

        :return: Generated scores including Accuracy, Precision, Recall, F-Score, and MCor as well as Numerical data
                 with total tweets predicted on, true positives, false positives, true negatives, false negatives
        :rtype: dict('Accuracy': float, 'Precision': float, 'Recall': float, 'F-Score': float, 'MCor': float,
                     'Numerical': dict('Total': int, 'True Positives': int, 'False Positives': int,
                     'True Negatives': int, 'False Negatives': int)
                    )
        """
        y = pd.read_csv(csv)['Label'].tolist()
        y1 = self.predict(csv)

        # Format: accuracy, precision, recall, f1, mcor, (total, tp, fp, tn, fn)
        scores = Utils.calculate_ml_measures(y1, y)

        if affect_parameter_scores:
            self.parameters.accuracy, self.parameters.precision, self.parameters.recall,\
                self.parameters.f_score, self.parameters.mcor = scores[0:5]

        keys = ['Accuracy', 'Precision', 'Recall', 'F-Score', 'MCor', 'Numerical']
        numerical = ['Total', 'True Positives', 'False Positives', 'True Negatives', 'False Negatives']
        score_dict = dict(zip(keys, scores))
        score_dict['Numerical'] = dict(zip(numerical, scores[5]))

        return score_dict

    def load_compile_test_model(self):
        """
        Loads a model from a previous file, compiles the model, and evaluates if selected.
        """

        self.model = nSC.load_saved_model(self.parameters.model_h5)
        self.compile_model()

        if self.parameters.evaluate_model:
            self.score = self.evaluate_model([self.data.test_text_input_ids,
                                              self.data.x_test_meta], self.data.y_test, [])

        return

    def get_callbacks(self):
        """
        Creates callbacks if requested. Supports early stopping and checkpoint callbacks.
        """

        cbs = []
        if self.parameters.early_stopping:
            # Set up early stopping callback
            cbs.append(nSC.create_early_stopping_callback('mcor', patience=self.parameters.early_stopping_patience))

        if self.parameters.checkpoint_model:
            # Set up checkpointing model
            cbs.append(nSC.create_model_checkpoint_callback(self.parameters.model_h5, monitor_stat='mcor',
                                                            mode='max'))

        return cbs

    def update_to_save_as_trained(self, nsc, text_input_length):
        # TODO ask Fedya why we have this
        self.parameters.custom_tokenizer = nsc.tokenizer
        self.parameters.custom_text_input_length = text_input_length
        self.parameters.train_data_csv = ''
        self.parameters.aug_data_csv = ''
        self.parameters.preload_train_data_dill = ''
        self.parameters.save_train_data_dill = ''
        self.parameters.load_to_predict = True

    def init_tpu(self):

        """ Initializes Google's Tensor Processing Unit.
        """

        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        self.tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

    @abstractmethod
    def build_model(self):
        pass
