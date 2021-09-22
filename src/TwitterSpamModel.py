import importlib
import tweepy
from bs4 import BeautifulSoup
import requests
import pandas as pd
import nltk
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
import pickle
import os
import NLPSentimentCalculations

NSC = NLPSentimentCalculations.NLPSentimentCalculations


class Version:
    def __init__(self,
                 comp_epochs=0,
                 saved_model_bin='',
                 original_data_csv='',
                 augmented_data_csv='',
                 early_stopping=False,
                 early_stopping_patience=0,
                 batch_size=128,
                 test_size=0.3,
                 features_to_train=None):

        if features_to_train is None:
            self.features_to_train = ['full_text']

        self.comp_epochs = comp_epochs
        self.saved_model_bin = saved_model_bin
        self.original_data_csv = original_data_csv
        self.augmented_data_csv = augmented_data_csv
        if augmented_data_csv:
            self.has_augmented = True
        else:
            self.has_augmented = False
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.batch_size = batch_size
        self.test_size = test_size
        self.accuracy = 0
        self.test_score = 0
        self.precision = 0
        self.recall = 0
        self.f_score = 0


class Model:
    def __init__(self):
        self.model = None
        self.version = None
        self.test_dataset = None
        self.nsc = NSC()

    def create_model(self, x_train_text_embeddings, x_test_text_embeddings, x_train_meta, x_test_meta,
                     glove_embedding_matrix, y_train, y_test, epochs, batch_size, saved_model_bin, early_stopping,
                     early_stopping_patience):

        spam_model = self.nsc.create_text_meta_model(glove_embedding_matrix,
                                                     len(x_train_meta.columns), len(x_train_text_embeddings[0]))

        # Print model summary
        spam_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        print(spam_model.summary())

        cbs = []
        if early_stopping:
            # Set up early stopping callback
            cbs.append(NSC.create_early_stopping_callback('acc', patience=early_stopping_patience))

            cbs.append(NSC.create_model_checkpoint_callback(saved_model_bin, monitor_stat='acc'))

        if len(x_train_meta.columns) < 1:
            train_input_layer = x_train_text_embeddings
            test_input_layer = x_test_text_embeddings
        else:
            train_input_layer = [x_train_text_embeddings, x_train_meta]
            test_input_layer = [x_test_text_embeddings, x_test_meta]

        history = spam_model.fit(x=train_input_layer, y=y_train, batch_size=batch_size,
                                 epochs=epochs, verbose=1, callbacks=cbs)

        if early_stopping and os.path.exists(saved_model_bin):
            spam_model = NSC.load_saved_model(saved_model_bin)

        score = spam_model.evaluate(x=test_input_layer, y=y_test, verbose=1, callbacks=[])

        print("Test Score:", score[0])
        print("Test Accuracy:", score[1])

        NSC.plot_model_history(history)

        self.model = spam_model

        return self.model, score
