import pandas as pd
import tensorflow as tf
from typing import List
import os
import NLPSentimentCalculations
import utilities
import ModelBase

Utils = utilities.Utils
NSC = NLPSentimentCalculations.NLPSentimentCalculations
ModelParameters = ModelBase.ModelParameters
ModelData = ModelBase.ModelData
ModelLearning = ModelBase.ModelLearning


class SpamModelData(ModelData):

    def __init__(self, nsc, base_data_csv, test_size, features_to_train, aug_data_csv=None, save_preload_binary='',
                 from_preload_binary=''):

        super().__init__(nsc, base_data_csv,
                         test_size,
                         features_to_train,
                         aug_data_csv)

        self.textless_features_to_train = [x for x in features_to_train if x != 'full_text']

        if from_preload_binary:
            self.load_data_from_binary(from_preload_binary)
        else:
            self.load_data_from_csv()

            if save_preload_binary:
                self.save_data_to_binary(save_preload_binary)

    def get_x_val_from_dataframe(self, x_val: pd.DataFrame) -> List[List[str]]:
        """
        Create an x_validation dataset from a dataframe, in a format ready to pass into model.predict

        :param x_val: Dataframe of tweets with self.features_to_train columns present
        :type x_val: pd.DataFrame

        :return: Data ready to be passed into the model for prediction
        :rtype: [x_val_text_embeddings, x_val_meta] or [x_val_text_embeddings]
        """

        x_val_text_embeddings = self.get_vectorized_text_tokens_from_val_dataframe(x_val)

        if len(self.textless_features_to_train) > 0:
            x_val_meta = x_val[self.textless_features_to_train]
            return [x_val_text_embeddings, x_val_meta]
        else:
            return [x_val_text_embeddings]

    def get_dataset_from_tweet_type(self, dataframe: pd.DataFrame):

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
        if 'full_text' in self.features_to_train:
            x_train_text_data = x_train['full_text']
            x_test_text_data = x_test['full_text']

        else:
            x_train_text_data = pd.DataFrame()
            x_test_text_data = pd.DataFrame()

        x_train_text_embeddings, \
        x_test_text_embeddings, \
        glove_embedding_matrix = self.get_vectorized_text_tokens_from_dataframes(x_train_text_data,
                                                                                 x_test_text_data)

        x_train_meta = x_train[self.textless_features_to_train]
        x_test_meta = x_test[self.textless_features_to_train]

        return x_train_text_embeddings, x_test_text_embeddings, x_train_meta, x_test_meta, \
               glove_embedding_matrix, y_train, y_test


class SpamModelLearning(ModelLearning):

    def __init__(self, model_params: ModelParameters, model_data: SpamModelData):

        super().__init__(model_params=model_params, model_data=model_data)

    def build_model(self):
        """
        Builds (trains) the model
        """

        if self.parameters.debug:
            tf.config.run_functions_eagerly(True)

        # Load previously saved model and test if requested
        if self.parameters.load_model and os.path.exists(self.parameters.saved_model_bin):
            self.load_compile_test_model()
            return

        self.model = self.data.nsc.create_spam_text_meta_model(self.data.glove_embedding_matrix,
                                                               len(self.data.x_train_meta.columns),
                                                               self.data.y_train.shape,
                                                               len(self.data.x_train_text_embeddings[0]))

        self.compile_model()

        cbs = self.get_callbacks()

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

        if self.parameters.evaluate_model:
            self.score = self.evaluate_model(test_input_layer, self.data.y_test, [])

        return
