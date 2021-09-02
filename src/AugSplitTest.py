import NLPSentimentCalculations
import NewsSentimentAnalysis
import pandas as pd
import math
import os


nsa = NewsSentimentAnalysis.TwitterManager()
nsc = NLPSentimentCalculations.NLPSentimentCalculations()

a = {"A": [1, 2, 3, 4, 5], "B": ['c', 'd', 'd', 'g', 'f'], "Label": [0, 0, 0, 1, 1], 'augmented': [1, 1, 0, 0, 0]}
a = pd.DataFrame(a)

nsc.keras_preprocessing(a[["A", "B"]], a['Label'], augmented_states=a['augmented'])


def test_bin_load():
    path = '../data/TestingData/save.pickle'
    nsa.initialize_twitter_spam_model(from_preprocess_binary=path)

def bin_save():
    path = '../data/TestingData/save.pickle'
    nsa.initialize_twitter_spam_model(to_preprocess_binary=path)