import nltk
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import pandas as pd
from nltk.tag import pos_tag
from nltk import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import re


class NLPSentimentCalculations:

    def __init__(self):

        self.classifier = None
        NLPSentimentCalculations.download_nltk_common()

    @staticmethod
    def download_nltk_common():

        nltk.download('wordnet')                     # For determining base words
        nltk.download('punkt')                       # Pretrained model to help with tokenizing
        nltk.download('averaged_perceptron_tagger')  # For determining word context

    def train_naivebayes_classifier(self, train_data):
        self.classifier = NaiveBayesClassifier.train(train_data)

    def test_classifier(self, test_data):
        print('Accuracy is:', classify.accuracy(self.classifier, test_data))
        print(self.classifier.show_most_informative_features(10))

    def classify_text(self, text):
        custom_tokens = NLPSentimentCalculations.sanitize_text(word_tokenize(text))
        return self.classifier.classify(dict([token, True] for token in custom_tokens))

    @staticmethod
    def get_all_words(all_tokens):
        for tokens in all_tokens:
            for token in tokens:
                yield token

    @staticmethod
    def get_clean_tokens(all_tokens, stop_words=()):

        cleaned_tokens = []
        for tokens in all_tokens:
            cleaned_tokens.append(NLPSentimentCalculations.sanitize_text(tokens, stop_words))
        return cleaned_tokens

    @staticmethod
    def get_freq_dist(all_tokens):
        all_words = NLPSentimentCalculations.get_all_words(all_tokens)
        return FreqDist(all_words)

    @staticmethod
    def get_basic_data_tag(all_tokens):
        for tokens in all_tokens:
            yield dict([token, True] for token in tokens)

    @staticmethod
    def get_basic_dataset(all_tokens, classifier_tag):
        token_tags = NLPSentimentCalculations.get_basic_data_tag(all_tokens)
        return [(class_dict, classifier_tag) for class_dict in token_tags]

    @staticmethod
    def sanitize_text(tweet_tokens, stop_words=()):

        cleaned_tokens = []

        for token, _ in pos_tag(tweet_tokens):

            token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                           '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
            token = re.sub("(@[A-Za-z0-9_]+)", "", token)

            wn = nltk.WordNetLemmatizer()
            token = wn.lemmatize(token)

            if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
                cleaned_tokens.append(token.lower())

        return cleaned_tokens

    # Discards all punctuation as classified by string.punctuation
    @staticmethod
    def remove_punctuation(text):
        return "".join([char for char in text if char not in string.punctuation])

    # Remove non-standard ASCII characters below char value 128
    @staticmethod
    def remove_bad_ascii(text):
        return "".join(i for i in text if ord(i) < 128)

    @staticmethod
    def collect_hashtags(text):
        return re.findall(r'#(\S*)\s?', text)

    @staticmethod
    def compute_tf_idf(tweets):

        # Dummy function to trick sklearn into taking token list
        def dummy_func(doc):
            return doc

        vectorizer = TfidfVectorizer(
            analyzer='word',
            tokenizer=dummy_func,
            preprocessor=dummy_func,
            token_pattern=None
        )

        vectors = vectorizer.fit_transform(tweets)
        feature_names = vectorizer.get_feature_names()

        dense = vectors.todense()
        denselist = dense.tolist()

        # Map TF-IDF results to dictionary
        tf_idf_list = []
        for tweetList in denselist:
            tf_idf_dict = dict.fromkeys(feature_names, 0)
            for i in range(0, len(feature_names)):
                tf_idf_dict[feature_names[i]] = tweetList[i]
            tf_idf_list.append(tf_idf_dict)

        return pd.Series(data=tf_idf_list)


def main():
    nsc = NLPSentimentCalculations()


if __name__ == '__main__':
    main()
