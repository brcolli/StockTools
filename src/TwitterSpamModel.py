import pandas as pd
import tensorflow as tf
import os
import NLPSentimentCalculations
import utilities
import ModelBase

Utils = utilities.Utils
NSC = NLPSentimentCalculations.NLPSentimentCalculations
ModelParameters = ModelBase.ModelParameters
ModelData = ModelBase.ModelData
ModelLearning = ModelBase.ModelLearning


class SpamModelParameters(ModelParameters):
    
    def __init__(self,
                 epochs=100,
                 saved_model_bin='',
                 early_stopping=False,
                 load_model=False,
                 early_stopping_patience=0,
                 batch_size=128):

        super().__init__(epochs,
                         saved_model_bin,
                         early_stopping,
                         load_model,
                         early_stopping_patience,
                         batch_size)


class SpamModelData(ModelData):

    def __init__(self, nsc, base_data_csv, test_size, features_to_train, aug_data_csv=None, save_preload_binary='',
                 from_preload_binary=''):

        super().__init__(nsc, base_data_csv,
                         test_size,
                         features_to_train,
                         aug_data_csv)

        if from_preload_binary:
            self.load_data_from_binary(from_preload_binary)
        else:
            self.load_data_from_csv()

            if save_preload_binary:
                self.save_data_to_binary(save_preload_binary)

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
        if 'full_text' in self.features_to_train:
            x_train_text_data = x_train['full_text']
            x_test_text_data = x_test['full_text']

            self.features_to_train.remove('full_text')
        else:
            x_train_text_data = pd.DataFrame()
            x_test_text_data = pd.DataFrame()

        # Clean the textual data
        x_train_text_clean = [NSC.sanitize_text_string(s) for s in list(x_train_text_data)]
        x_test_text_clean = [NSC.sanitize_text_string(s) for s in list(x_test_text_data)]

        # Initialize tokenizer on training data
        self.nsc.tokenizer.fit_on_texts(x_train_text_clean)

        # Create word vectors from tokens
        x_train_text_embeddings = self.nsc.keras_word_embeddings(x_train_text_clean)
        x_test_text_embeddings = self.nsc.keras_word_embeddings(x_test_text_clean)

        glove_embedding_matrix = self.nsc.create_glove_word_vectors()

        x_train_meta = x_train[self.features_to_train]
        x_test_meta = x_test[self.features_to_train]

        return x_train_text_embeddings, x_test_text_embeddings, x_train_meta, x_test_meta, \
               glove_embedding_matrix, y_train, y_test

    def load_data_from_csv(self):

        twitter_df = Utils.parse_json_tweet_data(self.base_data_csv, self.features_to_train)

        if 'augmented' not in twitter_df.columns:
            twitter_df['augmented'] = 0

        if self.aug_data_csv:
            aug_df = Utils.parse_json_tweet_data(self.aug_data_csv, self.features_to_train)
            if 'augmented' not in aug_df.columns:
                aug_df['augmented'] = 1

            twitter_df = pd.concat([twitter_df, aug_df])

        self.x_train_text_embeddings, self.x_test_text_embeddings, self.x_train_meta, self.x_test_meta, \
        self.glove_embedding_matrix,\
        self.y_train, self.y_test = self.get_dataset_from_tweet_spam(twitter_df)


class SpamModelLearning(ModelLearning):
    
    def __init__(self, model_params: SpamModelParameters, model_data: SpamModelData):

        super().__init__()

        self.model = tf.keras.models.Model()
        self.parameters = model_params
        self.data = model_data

    # TODO: Add function to test the model on provided csv of Tweets. Would be useful for validation later.
    # TODO: Add functions to calculate, store, and return aspects apart from accuracy like fscore, precision, and recall

    def evaluate_model(self, test_input_layer, test_labels, cbs):

        score = self.model.evaluate(x=test_input_layer, y=test_labels, verbose=1, callbacks=cbs)

        print("Test Score:", score[0])
        print("Test Accuracy:", score[1])

        return self.model, score

    def build_model(self):

        # Load previously saved model and test
        if self.parameters.load_model and os.path.exists(self.parameters.saved_model_bin):

            self.model = NSC.load_saved_model(self.parameters.saved_model_bin)

            return self.evaluate_model([self.data.x_test_text_embeddings, self.data.x_test_meta], self.data.y_test, [])

        spam_model = self.data.nsc.create_text_meta_model(self.data.glove_embedding_matrix,
                                                          len(self.data.x_train_meta.columns),
                                                          len(self.data.x_train_text_embeddings[0]))

        # Print model summary
        spam_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc',
                                                                                       tf.keras.metrics.Precision(),
                                                                                       tf.keras.metrics.Recall()])
        print(spam_model.summary())

        cbs = []
        if self.parameters.early_stopping:

            # Set up early stopping callback
            cbs.append(NSC.create_early_stopping_callback('acc', patience=self.parameters.early_stopping_patience))

            cbs.append(NSC.create_model_checkpoint_callback(self.parameters.saved_model_bin, monitor_stat='acc'))

        # Change input layer based on what style of features we are using
        if len(self.data.x_train_meta.columns) < 1:

            # Using only text features
            train_input_layer = self.data.x_train_text_embeddings
            test_input_layer = self.data.x_test_text_embeddings
        else:

            # Using text and meta features
            train_input_layer = [self.data.x_train_text_embeddings, self.data.x_train_meta]
            test_input_layer = [self.data.x_test_text_embeddings, self.data.x_test_meta]

        history = spam_model.fit(x=train_input_layer, y=self.data.y_train, batch_size=self.parameters.batch_size,
                                 epochs=self.parameters.epochs, verbose=1, callbacks=cbs)

        NSC.plot_model_history(history)

        self.model = spam_model

        return self.evaluate_model(test_input_layer, self.data.y_test, [])
