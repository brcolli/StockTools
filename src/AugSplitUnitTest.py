import unittest
import NLPSentimentCalculations
import NewsSentimentAnalysis
import pandas as pd
import math
import os

nsc = NLPSentimentCalculations.NLPSentimentCalculations()
nsa = NewsSentimentAnalysis.TwitterManager()


class MyTestCase(unittest.TestCase):
    def test_aug_split(self):
        a = {"A": [1, 2, 3, 4, 5], "B": ['c', 'd', 'd', 'g', 'f'], "Label": [0, 0, 0, 1, 1],
             'augmented': [1, 1, 0, 0, 0]}
        a = pd.DataFrame(a)
        x_train, x_test, y_train, y_test =\
            nsc.keras_preprocessing(a[["A", "B"]], a['Label'], augmented_states=a['augmented'])

        closest = math.floor((0.3 * len(a)) + 0.5) / len(a)

        self.assertEqual(len(x_train), len(y_train))
        self.assertEqual(len(x_test), len(y_test))
        self.assertAlmostEqual(closest, len(y_test)/len(a))

    def test_over_augmentation(self):
        a = {"A": [1, 2, 3, 4, 5], "B": ['c', 'd', 'd', 'g', 'f'], "Label": [0, 0, 0, 1, 1],
             'augmented': [1, 1, 1, 1, 1]}
        a = pd.DataFrame(a)
        x_train, x_test, y_train, y_test = \
            nsc.keras_preprocessing(a[["A", "B"]], a['Label'], augmented_states=a['augmented'])

        self.assertListEqual([x_train, x_test], [False]*2)

    def test_no_augmentation(self):
        a = {"A": [1, 2, 3, 4, 5], "B": ['c', 'd', 'd', 'g', 'f'], "Label": [0, 0, 0, 1, 1],
             'augmented': [1, 1, 1, 1, 1]}
        a = pd.DataFrame(a)
        x_train, x_test, y_train, y_test = \
            nsc.keras_preprocessing(a[["A", "B"]], a['Label'])

        closest = math.floor((0.3 * len(a)) + 0.5) / len(a)

        self.assertEqual(len(x_train), len(y_train))
        self.assertEqual(len(x_test), len(y_test))
        self.assertAlmostEqual(closest, len(y_test)/len(a))

    def test_bin_save(self):
        path = '../data/TestingData/save.pickle'
        nsa.initialize_twitter_spam_model(to_preprocess_binary=path)
        self.assertEqual(True, os.path.exists(path))

    def test_bin_load(self):
        path = '../data/TestingData/save.pickle'
        nsa.initialize_twitter_spam_model(from_preprocess_binary=path)
        self.assertEqual(True, True)



if __name__ == '__main__':
    unittest.main()
