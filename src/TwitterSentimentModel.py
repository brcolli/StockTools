import pandas as pd
import tensorflow as tf
import os
from typing import List
from NLPSentimentCalculations import NLPSentimentCalculations as nSC
from ModelBase import ModelParameters, ModelData, ModelLearning


class SentimentModelData(ModelData):

    def __init__(self, parameters: ModelParameters):

        super().__init__(parameters)

        if self.parameters.preload_train_data_dill:
            self.load_data_from_dill()
        else:
            self.load_data_from_csv()

            if self.parameters.save_train_data_dill:
                self.save_data_to_dill()

    def get_x_val_from_dataframe(self, x_val: pd.DataFrame) -> List[str]:
        """
        Create an x_validation dataset from a dataframe, in a format ready to pass into model.predict

        :param x_val: Dataframe of tweets with self.features_to_train columns present
        :type x_val: pd.DataFrame

        :return: Data ready to be passed into the model for prediction
        :rtype: x_val_text_embeddings
        """

        return self.get_vectorized_text_tokens_from_val_dataframe(x_val)

    def get_dataset_from_tweet_type(self, dataframe: pd.DataFrame):

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
        x_train_text_data = x_train['full_text']
        x_test_text_data = x_test['full_text']

        x_train_text_embeddings, \
        x_test_text_embeddings, \
        glove_embedding_matrix = self.get_vectorized_text_tokens_from_dataframes(x_train_text_data,
                                                                                 x_test_text_data)

        return x_train_text_embeddings, x_test_text_embeddings, glove_embedding_matrix, y_train, y_test


class SentimentModelLearning(ModelLearning):

    def __init__(self, model_params: ModelParameters, model_data: SentimentModelData):

        super().__init__(model_params=model_params, model_data=model_data)

    def build_model(self):
        """
        Builds (trains) the model

        :return: Model and score
        :rtype: tf.keras.Models.model, (float, float)
        """

        if self.parameters.debug:
            tf.config.run_functions_eagerly(True)

        # Load previously saved model and test
        if self.parameters.load_to_predict and os.path.exists(self.parameters.model_h5):
            # @TODO This needs to be fixed because {data.x_test_text_embeddings}, etc. does not get preprocessed if
            # @TODO model is being loaded from h5.
            self.load_compile_test_model()
            return

        self.model = self.data.nsc.create_sentiment_text_model(self.data.glove_embedding_matrix,
                                                               self.data.y_train.shape,
                                                               len(self.data.x_train_text_embeddings[0]))

        self.compile_model()

        cbs = self.get_callbacks()

        history = self.model.fit(x=self.data.x_train_text_embeddings, y=self.data.y_train,
                                 batch_size=self.parameters.batch_size,
                                 epochs=self.parameters.epochs, verbose=1, callbacks=cbs)

        nSC.plot_model_history(history)

        if self.parameters.evaluate_model:
            self.score = self.evaluate_model(self.data.x_test_text_embeddings, self.data.y_test, [])

        return
