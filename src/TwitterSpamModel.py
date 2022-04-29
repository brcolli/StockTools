import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import os
import NLPSentimentCalculations
import utilities
import ModelBase
from typing import Tuple

Utils = utilities.Utils
NSC = NLPSentimentCalculations.NLPSentimentCalculations
ModelParameters = ModelBase.ModelParameters
ModelData = ModelBase.ModelData
ModelLearning = ModelBase.ModelLearning

# These have to be here and not in ModelParameters because they need to know about NSC functions
Metrics = ['acc', NSC.precision, NSC.recall, NSC.mcor,
           tfa.metrics.FBetaScore(num_classes=2, average='weighted', beta=1.0, name='fbeta')]
MetricsKeys = ['acc', 'precision', 'recall', 'mcor', 'fbeta']
MetricsDict = dict(zip(MetricsKeys, Metrics))


# Class to handle model settings, loading settings, and data file paths. Currently identical to ModelBase.py version.
class SpamModelParameters(ModelParameters):

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
                 model_h5=''):

        dict_args = locals()
        dict_args.pop('self')
        dict_args.pop('__class__')
        super().__init__(**dict_args)


class SpamModelData(ModelData):

    def __init__(self, parameters: SpamModelParameters):

        super().__init__(parameters)

        if self.parameters.features_to_train is None:
            self.parameters.features_to_train = ['full_text']

        self.parameters.textless_features_to_train = [x for x in self.parameters.features_to_train if x != 'full_text']

        # Load Data necessary for training mode
        if self.parameters.load_to_train:
            # Preload train data from a dill
            if parameters.preload_train_data_dill:
                self.load_data_from_dill()

            # Load data normally (from CSVs)
            else:
                self.load_data_from_csv()

                # Save train data to a dill for preloading next time
                if self.parameters.save_train_data_dill:
                    self.save_data_to_dill()

        # Nothing needs to be done if we are loading for prediction only mode
        else:
            pass

    def get_x_val_from_csv(self, csv):
        """
        Loads an x_validation dataset from a csv, in a format ready to pass into model.predict

        :param csv: Path to csv file containing a saved dataframe of tweets with self.features_to_train columns present
        :type csv: str

        :return: Data ready to be passed into the model for prediction
        :rtype: [x_val_text_embeddings, x_val_meta] or x_val_text_embeddings
        """

        df = Utils.parse_json_tweet_data_from_csv(csv, self.parameters.features_to_train)
        return self.get_x_val_from_dataframe(df)

    def get_x_val_from_dataframe(self, x_val):
        """
        Create an x_validation dataset from a dataframe, in a format ready to pass into model.predict

        :param x_val: Dataframe of tweets with self.features_to_train columns present
        :type x_val: pd.DataFrame

        :return: Data ready to be passed into the model for prediction
        :rtype: [x_val_text_embeddings, x_val_meta] or x_val_text_embeddings
        """
        if 'full_text' in self.parameters.features_to_train:
            x_val_text_data = x_val['full_text']
        else:
            x_val_text_data = pd.DataFrame()

        # Sanitizes textual data
        x_val_text_clean = [NSC.sanitize_text_string(s) for s in list(x_val_text_data)]

        # Vectorizes textual data
        _, x_val_text_embeddings = self.nsc.keras_word_embeddings(x_val_text_clean, self.text_input_length)

        if len(self.parameters.textless_features_to_train) > 0:
            x_val_meta = x_val[self.parameters.textless_features_to_train]
            return [x_val_text_embeddings, x_val_meta]
        else:
            return x_val_text_embeddings

    def get_dataset_from_tweet_spam(self, dataframe):

        """Converts the text feature to a dataset of labeled unigrams and bigrams.

        :param dataframe: A dataframe containing the text key with all the text features to parse
        :type dataframe: :class:`pandas.core.frame.DataFrame`

        :return: A list of tuples of dictionaries of feature mappings to classifiers
        :rtype: list(dict(str-> bool), str)
        """

        x_train, x_test, y_train, y_test = self.nsc.keras_preprocessing(dataframe[self.parameters.features_to_train],
                                                                        dataframe['Label'],
                                                                        augmented_states=dataframe['augmented'],
                                                                        test_size=self.parameters.test_size)

        if x_train is False:
            print("Train test failed due to over augmentation")
            return False

        # Split into text and meta data
        if 'full_text' in self.parameters.features_to_train:
            x_train_text_data = x_train['full_text']
            x_test_text_data = x_test['full_text']

        else:
            x_train_text_data = pd.DataFrame()
            x_test_text_data = pd.DataFrame()

        # Clean the textual data
        x_train_text_clean = [NSC.sanitize_text_string(s) for s in list(x_train_text_data)]
        x_test_text_clean = [NSC.sanitize_text_string(s) for s in list(x_test_text_data)]

        # Initialize tokenizer on training data
        self.nsc.tokenizer.fit_on_texts(x_train_text_clean)

        # Create word vectors from tokens
        self.text_input_length, x_train_text_embeddings = self.nsc.keras_word_embeddings(x_train_text_clean)
        _, x_test_text_embeddings = self.nsc.keras_word_embeddings(x_test_text_clean, self.text_input_length)

        glove_embedding_matrix = self.nsc.create_glove_word_vectors()

        x_train_meta = x_train[self.parameters.textless_features_to_train]
        x_test_meta = x_test[self.parameters.textless_features_to_train]

        return x_train_text_embeddings, x_test_text_embeddings, x_train_meta, x_test_meta, \
               glove_embedding_matrix, y_train, y_test

    def load_data_from_csv(self):
        """
        Loads twitter dataframe from csv and calls self.get_dataset_from_tweet_sentiment
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

        self.x_train_text_embeddings, self.x_test_text_embeddings, self.x_train_meta, self.x_test_meta, \
        self.glove_embedding_matrix, self.y_train, self.y_test = self.get_dataset_from_tweet_spam(
            twitter_df)


class SpamModelLearning(ModelLearning):

    def __init__(self, model_params: SpamModelParameters, model_data: SpamModelData):

        super().__init__()

        self.model = tf.keras.models.Model()
        self.parameters = model_params
        self.data = model_data
        self.metrics = Metrics

    def compile_model(self):

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.parameters.learning_rate)
        self.model.compile(loss='binary_crossentropy',
                           optimizer=optimizer,
                           metrics=self.metrics)
        print(self.model.summary())  # Print model summary

    def raw_predict_csv(self, csv):
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

    def raw_predict_tweets(self, tweet_df):
        x_val = self.data.get_x_val_from_dataframe(tweet_df)
        return self.model.predict(x_val)

    def raw_predict_from_x_val(self, x_val):
        return self.model.predict(x_val)

    def predict(self, csv='', tweet_df=None):
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

        # Use the highest softmax probability as the label (0 or 1)
        return [max(range(len(y1)), key=y1.__getitem__) for y1 in y]

    def predict_and_score(self, csv, affect_parameter_scores=False):
        """
        Predicts Tweet labels from a csv of Tweets. Then scores those predictions using labels provided in the csv. CSV
        must be a saved dataframe of tweets with all the columns in self.data.features_to_train + 'Label' present.

        :param csv: Filepath to dataframe of tweets with self.data.features_to_train and 'Label' columns
        :type csv: str
        :param affect_parameter_scores: Whether or not to write the generated scores to self.parameters
        :type affect_parameter_scores: bool

        :return: Generated scores including Accuracy, Precision, Recall, F-Score, and MCor as well as Numerical data with
                total tweets predicted on, true positives, false positives, true negatives, false negatives
        :rtype: dict(
                    'Accuracy': float, 'Precision': float, 'Recall': float, 'F-Score': float, 'MCor': float,
                    'Numerical': dict('Total': int, 'True Positives': int, 'False Positives': int, 'True Negatives': int,
                        'False Negatives': int)
                    )
        """
        y = pd.read_csv(csv)['Label'].tolist()
        y1 = self.predict(csv)

        # Format: accuracy, precision, recall, f1, mcor, (total, tp, fp, tn, fn)
        scores = Utils.calculate_ml_measures(y1, y)

        if affect_parameter_scores:
            self.parameters.accuracy, self.parameters.precision, self.parameters.recall, \
            self.parameters.f_score, self.parameters.mcor = scores[0:5]

        keys = ['Accuracy', 'Precision', 'Recall', 'F-Score', 'MCor', 'Numerical']
        numerical = ['Total', 'True Positives', 'False Positives', 'True Negatives', 'False Negatives']
        score_dict = dict(zip(keys, scores))
        score_dict['Numerical'] = dict(zip(numerical, scores[5]))

        return score_dict

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

        self.parameters.trained = True

        return score

    def build_model(self) -> None:
        """
        Builds (trains) the model
        """

        if self.parameters.debug:
            tf.config.run_functions_eagerly(True)

        # Load previously saved model and test if requested
        if self.parameters.load_to_predict and os.path.exists(self.parameters.h5):

            self.model = NSC.load_saved_model(self.parameters.h5)
            self.compile_model()

            # @TODO This needs to be fixed because {data.x_test_text_embeddings}, etc. does not get preprocessed if
            # @TODO model is being loaded from h5.
            if self.parameters.evaluate_model_on_load:
                print("\nEvaluation not currently supported for loading model in predict mode\n")
                # self.score = self.evaluate_model([self.data.x_test_text_embeddings,
                #                                   self.data.x_test_meta], self.data.y_test, [])

            return

        self.model = self.data.nsc.create_spam_text_meta_model(self.data.glove_embedding_matrix,
                                                               len(self.data.x_train_meta.columns),
                                                               self.data.y_train.shape,
                                                               len(self.data.x_train_text_embeddings[0]))

        self.compile_model()

        cbs = []
        if self.parameters.early_stopping:
            # Set up early stopping callback
            cbs.append(NSC.create_early_stopping_callback('mcor', patience=self.parameters.early_stopping_patience))

        if self.parameters.checkpoint_model:
            # Set up checkpointing model
            cbs.append(NSC.create_model_checkpoint_callback(self.parameters.h5, monitor_stat='mcor',
                                                            mode='max'))

        # Change input layer based on what style of features we are using
        if len(self.data.x_train_meta.columns) < 1:

            # Using only text features
            train_input_layer = self.data.x_train_text_embeddings
            test_input_layer = self.data.x_test_text_embeddings
        else:

            # Using text and meta features
            train_input_layer = [self.data.x_train_text_embeddings, self.data.x_train_meta]
            test_input_layer = [self.data.x_test_text_embeddings, self.data.x_test_meta]

        history = self.model.fit(x=train_input_layer, y=self.data.y_train, batch_size=self.parameters.batch_size,
                                 epochs=self.parameters.epochs, verbose=1, callbacks=cbs)

        NSC.plot_model_history(history)

        if self.parameters.evaluate_model_on_load:
            self.parameters.score = self.evaluate_model(test_input_layer, self.data.y_test, [])

        return

    # def save_trained_model_to_h5(self):
    #     print("Saving model")
    #     self.model.save(self.parameters.h5)
