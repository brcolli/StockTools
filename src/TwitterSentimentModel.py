import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import os
import NLPSentimentCalculations
import utilities
import ModelBase

Utils = utilities.Utils
NSC = NLPSentimentCalculations.NLPSentimentCalculations
ModelParameters = ModelBase.ModelParameters
ModelData = ModelBase.ModelData
ModelLearning = ModelBase.ModelLearning
Metrics = ['acc', NSC.precision, NSC.recall, NSC.mcor,
           tfa.metrics.FBetaScore(num_classes=2, average='weighted', beta=1.0, name='fbeta')]
MetricsKeys = ['acc', 'precision', 'recall', 'mcor', 'fbeta']
MetricsDict = dict(zip(MetricsKeys, Metrics))


class SentimentModelParameters(ModelParameters):

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
        super().__init__(learning_rate,
                         epochs,
                         saved_model_bin,
                         early_stopping,
                         checkpoint_model,
                         load_model,
                         early_stopping_patience,
                         batch_size,
                         trained,
                         debug)


class SentimentModelData(ModelData):

    def __init__(self, nsc, base_data_csv, test_size, aug_data_csv=None, save_preload_binary='',
                 from_preload_binary=''):

        super().__init__(nsc=nsc, base_data_csv=base_data_csv,
                         test_size=test_size,
                         features_to_train=None,
                         aug_data_csv=aug_data_csv)

        if from_preload_binary:
            self.load_data_from_binary(from_preload_binary)
        else:
            self.load_data_from_csv()

            if save_preload_binary:
                self.save_data_to_binary(save_preload_binary)

    def get_x_val_from_csv(self, csv):
        """
        Loads an x_validation dataset from a csv, in a format ready to pass into model.predict

        :param csv: Path to csv file containing a saved dataframe of tweets with self.features_to_train columns present
        :type csv: str

        :return: Data ready to be passed into the model for prediction
        :rtype: [x_val_text_embeddings, x_val_meta] or x_val_text_embeddings
        """

        df = Utils.parse_json_tweet_data(csv, self.features_to_train)
        return self.get_x_val_from_dataframe(df)

    def get_x_val_from_dataframe(self, x_val):
        """
        Create an x_validation dataset from a dataframe, in a format ready to pass into model.predict

        :param x_val: Dataframe of tweets with self.features_to_train columns present
        :type x_val: pd.DataFrame

        :return: Data ready to be passed into the model for prediction
        :rtype: [x_val_text_embeddings, x_val_meta] or x_val_text_embeddings
        """

        x_val_text_data = x_val['full_text']

        # Sanitizes textual data
        x_val_text_clean = [NSC.sanitize_text_string(s) for s in list(x_val_text_data)]

        # Vectorizes textual data
        _, x_val_text_embeddings = self.nsc.keras_word_embeddings(x_val_text_clean, self.text_input_length)

        return x_val_text_embeddings

    def get_dataset_from_tweet_spam(self, dataframe):

        """Converts the text feature to a dataset of labeled unigrams and bigrams.

        :param dataframe: A dataframe containing the text key with all the text features to parse
        :type dataframe: :class:`pandas.core.frame.DataFrame`

        :return: A list of tuples of dictionaries of feature mappings to classifiers
        :rtype: list(dict(str-> bool), str)
        """

        x_train, x_test, y_train, y_test = self.nsc.keras_preprocessing(dataframe[self.features_to_train],
                                                                        dataframe['Label'],
                                                                        augmented_states=dataframe['augmented'],
                                                                        test_size=self.test_size)

        if x_train is False:
            print("Train test failed due to over augmentation")
            return False

        # Split into text and meta data
        x_train_text_data = x_train['full_text']
        x_test_text_data = x_test['full_text']

        # Clean the textual data
        x_train_text_clean = [NSC.sanitize_text_string(s) for s in list(x_train_text_data)]
        x_test_text_clean = [NSC.sanitize_text_string(s) for s in list(x_test_text_data)]

        # Initialize tokenizer on training data
        self.nsc.tokenizer.fit_on_texts(x_train_text_clean)

        # Create word vectors from tokens
        self.text_input_length, x_train_text_embeddings = self.nsc.keras_word_embeddings(x_train_text_clean)
        _, x_test_text_embeddings = self.nsc.keras_word_embeddings(x_test_text_clean, self.text_input_length)

        glove_embedding_matrix = self.nsc.create_glove_word_vectors()

        return x_train_text_embeddings, x_test_text_embeddings, glove_embedding_matrix, y_train, y_test

    def load_data_from_csv(self):
        """
        Loads twitter dataframe from csv and calls self.get_dataset_from_tweet_spam
        """

        twitter_df = Utils.parse_json_tweet_data(self.base_data_csv, self.features_to_train)

        if 'augmented' not in twitter_df.columns:
            twitter_df['augmented'] = 0

        if self.aug_data_csv:
            aug_df = Utils.parse_json_tweet_data(self.aug_data_csv, self.features_to_train)
            if 'augmented' not in aug_df.columns:
                aug_df['augmented'] = 1

            twitter_df = pd.concat([twitter_df, aug_df])

        self.x_train_text_embeddings, self.x_test_text_embeddings, self.glove_embedding_matrix, \
        self.y_train, self.y_test = self.get_dataset_from_tweet_spam(twitter_df)


class SentimentModelLearning(ModelLearning):

    def __init__(self, model_params: SentimentModelParameters, model_data: SentimentModelData):

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

    # TODO: Add function to test the model on provided csv of Tweets. Would be useful for validation later.
    # TODO: Add functions to calculate, store, and return aspects apart from accuracy like fscore, precision, and recall

    def raw_predict(self, csv):
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

    def predict(self, csv):
        """
        Predicts Tweet labels from a csv of Tweets. CSV must be a saved dataframe of tweets with all the
        columns in self.data.features_to_train present. Each prediction is either a -1 (not marked), 0 (clean), or
        1 (spam).

        :param csv: Filepath to dataframe of tweets with self.data.features_to_train columns
        :type csv: str

        :return: Prediction for each tweet
        :rtype: [int] where int is -1, 0, or 1
        """
        # Get the raw softmax probability predictions
        y = self.raw_predict(csv)

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
            self.parameters.accuracy, self.parameters.precision, self.parameters.recall,\
            self.parameters.f_score, self.parameters.mcor = scores[0:5]

        keys = ['Accuracy', 'Precision', 'Recall', 'F-Score', 'MCor', 'Numerical']
        numerical = ['Total', 'True Positives', 'False Positives', 'True Negatives', 'False Negatives']
        score_dict = dict(zip(keys, scores))
        score_dict['Numerical'] = dict(zip(numerical, scores[4]))

        return score_dict

    def evaluate_model(self, test_input_layer, test_labels, cbs):
        """
        Evaluate model using tensorflow methods

        :param test_input_layer: Model input layer being [x_val_text_embeddings, x_val_meta] or x_val_text_embeddings
        :type test_input_layer: list
        :param test_labels: Tweet labels
        :type test_labels: list(int)
        :param cbs: List of callbacks
        :type cbs: list(func)

        :return: Model and score
        :rtype: tf.keras.Models.model, (float, float)
        """

        score = self.model.evaluate(x=test_input_layer, y=test_labels, verbose=1, callbacks=cbs)

        print("Test Score:", score[0])
        print("Test Accuracy:", score[1])

        self.parameters.trained = True

        return self.model, score

    def build_model(self):
        """
        Builds (trains) the model

        :return: Model and score
        :rtype: tf.keras.Models.model, (float, float)
        """

        if self.parameters.debug:
            tf.config.run_functions_eagerly(True)

        # Load previously saved model and test
        if self.parameters.load_model and os.path.exists(self.parameters.saved_model_bin):
            self.model = NSC.load_saved_model(self.parameters.saved_model_bin)
            self.compile_model()

            return self.evaluate_model(self.data.x_test_text_embeddings, self.data.y_test, [])

        self.model = self.data.nsc.create_sentiment_text_model(self.data.glove_embedding_matrix,
                                                               self.data.y_train.shape,
                                                               len(self.data.x_train_text_embeddings[0]))

        self.compile_model()

        cbs = []
        if self.parameters.early_stopping:
            # Set up early stopping callback
            cbs.append(NSC.create_early_stopping_callback('mcor', patience=self.parameters.early_stopping_patience))

        if self.parameters.checkpoint_model:
            # Set up checkpointing model
            cbs.append(NSC.create_model_checkpoint_callback(self.parameters.saved_model_bin, monitor_stat='mcor',
                                                            mode='max'))

        history = self.model.fit(x=self.data.x_train_text_embeddings, y=self.data.y_train, batch_size=self.parameters.batch_size,
                                 epochs=self.parameters.epochs, verbose=1, callbacks=cbs)

        NSC.plot_model_history(history)

        return self.evaluate_model(self.data.x_test_text_embeddings, self.data.y_test, [])
