from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from imblearn.over_sampling import BorderlineSMOTE
import eda
import pandas as pd
import BotometerRequests
import nltk
import utilities
import random
import math
import copy

Utils = utilities.Utils()
nltk.download('wordnet')


class TweetEDA:
    """
    Used to augment tweets using eda.py augmentation techniques, also has remove url method
    """

    def __init__(self):
        pass

    def remove_urls(self, json: str or dict) -> str:
        """
        A useful method to clean urls from Tweet text. Takes a tweet object and uses different information to remove
        urls. Then returns the tweet text without urls.

        :param json: Tweet object json, either as a str or dict
        :type json: str or dict

        :return: The text of the Tweet without any urls in it
        :rtype: str
        """

        json = Utils.safe_str_to_dict(json)
        text_string = self.text_from_tweet(json)

        # Checks for urls provided in the Tweet object
        if 'urls' in json.keys():
            if 'url' in json['urls'].keys():
                text_string = text_string.replace(json['urls']['url'], '')

        # Removes continuous segments which start with http, possible to add more keywords
        url_start = 'http'
        while text_string.find(url_start) != -1:
            found = text_string.find(url_start)
            remove = len(text_string)
            for i in range(found, len(text_string)):
                if text_string[i] in ' \n\t':
                    remove = i
                    break

            text_string = text_string[0:found] + text_string[remove:]

        return text_string

    @staticmethod
    def multi_method_augmentation(sentence: str, alpha_sr: float, alpha_ri: float, alpha_rs: float, alpha_rd: float,
                                  total_augment: float, order=None):
        if order is None:
            order = ['SR', 'RI', 'RS', 'RD']

        sentence = eda.get_only_chars(sentence)
        words = sentence.split(' ')
        words = [word for word in words if word != ' ']

        aug_values = [alpha_sr, alpha_ri, alpha_rs, alpha_rd]
        s = sum(aug_values)
        aug_values = [v / s for v in aug_values]
        aug_values = [x * total_augment for x in aug_values]
        aug_values = [max(1, int(len(words) * x)) for x in aug_values[0:3]] + [aug_values[3]]
        funcs = [eda.synonym_replacement, eda.random_insertion, eda.random_swap, eda.random_deletion]
        func_val_dicts = [{'Func': f, 'Val': v} for f, v in zip(funcs, aug_values)]
        operation_dict = dict(zip(['SR', 'RI', 'RS', 'RD'], func_val_dicts))

        for o in order:
            words = operation_dict[o]['Func'](words, operation_dict[o]['Val'])

        res = ' '.join(words)
        return res

    @staticmethod
    def augment_sentences(sentences: [str], alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, alpha_rd=0.1,
                          num_aug_per_sentence=9) -> [str]:
        """
        Augments sentences using eda.py

        :param sentences: Tweet texts to augment
        :type sentences: [str]

        :param alpha_sr: The percentage of words to replace with synonyms in a sentence. 0 to not use this method.
        :type alpha_sr: float in (0, 1)

        :param alpha_ri: The amount of random insertions to make as a percentage of the number of words in a sentence.
                            0 to not use this method.
        :type alpha_ri: float in (0, 1)

        :param alpha_rs: The amount of random swaps to make as a percentage of the number of words in a sentence. 0 to
                            not use this method.
        :type alpha_rs: float in (0, 1)

        :param alpha_rd: The chance any single word gets randomly deleted in a sentence. 0 to not use this method.
        :type alpha_rd: float in (0, 1)

        :param num_aug_per_sentence: Number of augmented strings to make for each original
        :type num_aug_per_sentence: int

        :return: List of augmented strings segmented into lists by original string
                Ex: sentences = ["abc", "def"] num_aug = 3
                return: [["abc1", "abc2", "abc3"], ["def1", "def2", "def3"]]
        :rtype: [[str]]
        """
        augmented_sentences = [eda.eda(s, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd,
                                       num_aug=num_aug_per_sentence)[:-1] for s in sentences]
        return augmented_sentences

    @staticmethod
    def text_from_tweet(string_json: str or dict) -> str:
        """
        Extracts the Tweet text from a str or dictionary Tweet json objects. Necessary because some Tweets have the key
        'full_text', while others have 'text'.

        :param string_json: Tweet object json, either as a str or dict
        :type string_json: str or dict

        :return: Text of the Tweet
        :rtype: str
        """
        string_json = Utils.safe_str_to_dict(string_json)
        try:
            txt = string_json['full_text']
        except KeyError:
            try:
                txt = string_json['text']
            except KeyError:
                txt = ''

        return txt

    @staticmethod
    def replace_text_in_json(string_json: str or dict, text: str) -> dict:
        """
        Replaces the existing text of a Tweet json object with a new text.

        :param string_json: Tweet object json, either as a str or dict
        :type string_json: str or dict
        :param text: Text to insert instead of the current text
        :type text: str

        :return: Tweet object json, as a dict
        :rtype: dict
        """
        dictionary = Utils.safe_str_to_dict(string_json)
        keys = dictionary.keys()
        if 'full_text' in keys:
            dictionary['full_text'] = text
        elif 'text' in keys:
            dictionary['text'] = text
        return dictionary

    def remove_urls_from_objects(self, objects: [str] or [dict]) -> [dict]:
        """
        Wrapper method of self.remove_urls for a list of Tweet objects

        :param objects: List of Tweet objects either as str or dict
        :type objects: [str] or [dict]

        :return: List of Tweet objects with urls removed from the Tweet texts
        :rtype: [dict]
        """
        return [self.replace_text_in_json(o, self.remove_urls(o)) for o in objects]

    def remove_urls_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Wrapper method of self.remove_urls for a dataframe of Tweet data which includes the Tweet objects under key
        'json'

        :param df: Dataframe of Tweet data, required key: 'json'
        :type df: pd.DataFrame

        :return: Dataframe with urls removed from Tweet object texts. Tweet objects serialized as dictionaries if were
                    previously strings
        :rtype: pd.DataFrame
        """
        df['json'] = self.remove_urls_from_objects(df['json'])
        return df

    @staticmethod
    def score_new_objects(df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a dataframe of newly augmented tweets and scores them, then inserts the botscores and returns dataframe

        :param df: Dataframe with keys ['Tweet id', 'text', 'json'] obtained from self.wrapper
        :type df: pd.DataFrame

        :return: Dataframe with scored botscore
        :rtype: pd.DataFrame
        """
        br = BotometerRequests.BotometerRequests()
        scored = br.wrapper(from_dataframe=df, lite_tweet_request=True)
        df['botscore'] = scored['botscore']
        return df

    def random_multi_wrapper(self, sentences, aug_per_sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, alpha_rd=0.1,
                             total_mod=0.5):
        rand_values = lambda v: [random.uniform(0.5 * x, 1.5 * x) for x in v]
        vals = [alpha_sr, alpha_ri, alpha_rs, alpha_rd, total_mod]
        order = ['SR', 'RI', 'RS', 'RD']
        augmented = []

        for s in sentences:
            new = []
            for _ in range(aug_per_sentence):
                sr, ri, rs, rd, tm = rand_values(vals)
                random.shuffle(order)
                new.append(self.multi_method_augmentation(s, sr, ri, rs, rd, tm, order=order))
            augmented.append(new.copy())

        return augmented

    def wrapper(self, entry_data=pd.DataFrame(), score_new_objects=False, keep_ua_data=True, total_aug_to_make=100,
                general_alpha=0, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, alpha_rd=0.1, increment_alpha=0,
                random_multi=False, multi_total=0.5, to_file='', from_file='') -> pd.DataFrame:
        """
        A method to get EDA augmented TweetObjects and their botometer lite scores

        :param entry_data: The baseline data formatted as a df. Required keys: ['Tweet id', 'json']
        :type entry_data: pd.DataFrame

        :param score_new_objects: Whether or not to score the newly augmented Tweets with Botometer Lite
        :type score_new_objects: bool

        :param total_aug_to_make: How many new augmented sentences to make in total: ie How many augmented data points
                                    to create.
        :type total_aug_to_make: int

        :param general_alpha: A shortcut to passing individual alphas for each type of augmentation. Sets all alphas to
                                this value.
        :type general_alpha: float in (0, 1)

        :param alpha_sr: The percentage of words to replace with synonyms in a sentence. 0 to not use this method.
        :type alpha_sr: float in (0, 1)

        :param alpha_ri: The amount of random insertions to make as a percentage of the number of words in a sentence.
                            0 to not use this method.
        :type alpha_ri: float in (0, 1)

        :param alpha_rs: The amount of random swaps to make as a percentage of the number of words in a sentence. 0 to
                            not use this method.
        :type alpha_rs: float in (0, 1)

        :param alpha_rd: The chance any single word gets randomly deleted in a sentence. 0 to not use this method.
        :type alpha_rd: float in (0, 1)

        :param increment_alpha: The amount by which to increase or decrease the alpha for every round of augmentation.
                                If this value is not zero, no sentence will be augmented with the same alpha values
                                twice. Notice that alpha should stay in range (0, 1) so #rounds x increment_alpha should
                                not put alpha out of range.
        :type increment_alpha: float in (-1, 1)

        :return: Dataframe with the keys: ['Tweet id', 'text', 'json', 'botscore'(optional)]. Tweet ids for augmented
                tweets are generated as (Original_id * 100) + n. Text has the augmented text of the tweet. json has the
                augmented text reinserted into the Tweet objects. Storing each augmented tweet reinserted into the
                original tweet object takes significantly more space than storing only augmented text.
        :rtype: pd.DataFrame
        """

        if from_file:
            entry_data = pd.read_csv(from_file)

        if general_alpha > 0:
            alpha_sr, alpha_ri, alpha_rs, alpha_rd = [general_alpha] * 4

        non_zero_techniques = [x for x in (alpha_sr, alpha_ri, alpha_rs, alpha_rd) if x != 0]

        ids = entry_data['Tweet id'].tolist()
        tweet_objects = [self.replace_text_in_json(o, self.remove_urls(o)) for o in entry_data['json']]
        sentences = [self.text_from_tweet(t) for t in tweet_objects]
        aug_per_sentence = math.ceil(total_aug_to_make / len(entry_data))

        if random_multi:
            aug_s = self.random_multi_wrapper(sentences, aug_per_sentence, alpha_sr, alpha_ri, alpha_rs, alpha_rd,
                                              multi_total)

        else:
            if increment_alpha != 0:
                if (((aug_per_sentence - 1) * increment_alpha) + max(non_zero_techniques) > 1) or \
                        (((aug_per_sentence - 1) * increment_alpha) + min(non_zero_techniques) < 0):
                    raise Exception("Increment alpha puts alpha out of range (0, 1)")

                aug_s = []
                original = [alpha_sr, alpha_ri, alpha_rs, alpha_rd]

                for s in sentences:
                    s_augs = []
                    for _ in range(aug_per_sentence):
                        s_augs.append(eda.eda(s, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd,
                                              num_aug=1)[0])

                        alpha_sr, alpha_ri, alpha_rs, alpha_rd = [x + increment_alpha if x != 0 else 0 for x in (alpha_sr,
                                                                                                                 alpha_ri,
                                                                                                                 alpha_rs,
                                                                                                                 alpha_rd)]
                    alpha_sr, alpha_ri, alpha_rs, alpha_rd = original
                    aug_s.append(s_augs)
            else:
                aug_s = self.augment_sentences(sentences, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs,
                                               alpha_rd=alpha_rd, num_aug_per_sentence=aug_per_sentence)

        original_index_references = [i for i in range(len(sentences)) for _ in range(aug_per_sentence)]
        new_index_reference = [i for i in range(aug_per_sentence)] * len(aug_s)

        new_ids = [(ids[a] * 100) + b + 1 for a, b in zip(original_index_references, new_index_reference)]
        new_sentences = Utils.flatten(aug_s)

        new_objects = [copy.deepcopy(self.replace_text_in_json(tweet_objects[original_index_references[i]],
                                                               new_sentences[i])) for i in range(len(new_sentences))]

        new_df = [entry_data.iloc[i] for i in original_index_references]
        new_df = pd.DataFrame(new_df)
        new_df['Tweet id'] = new_ids
        new_df['json'] = new_objects
        new_df['aug sentences'] = new_sentences
        new_df['augmented'] = 1
        remainder = len(new_sentences) - total_aug_to_make

        if remainder != 0:
            remove_indices = random.sample(range(len(new_df)), remainder)
            keep_indices = list(set(remove_indices) ^ set(range(len(new_df))))
            new_df = pd.DataFrame([new_df.iloc[i] for i in keep_indices])

        if not keep_ua_data:
            new_df = new_df[['Tweet id', 'Label', 'json', 'aug sentences', 'augmented']]

        new_df = new_df.reset_index(drop=True)

        if score_new_objects:
            new_df = self.score_new_objects(new_df)

        if to_file:
            try:
                new_df.to_csv(to_file, index=False)
            except:
                print(f'File {to_file} not found, dataframe not written')

        return new_df


class NumericalDataAugmentationManager:

    @staticmethod
    def shuffle_meta_scores(df, meta_headers, group=True):

        if group:

            spam_df = df[df['Label'] == 1].reset_index(drop=True)
            y = spam_df[meta_headers].sample(frac=1)
            spam_df[meta_headers] = y.reset_index(drop=True)

            ham_df = df[df['Label'] == 0].reset_index(drop=True)
            ham_df[meta_headers] = ham_df[meta_headers].sample(frac=1).reset_index(drop=True)
        else:

            spam_df = df[df['Label'] == 1].reset_index(drop=True)
            y = spam_df[meta_headers].apply(lambda m: m.sample(frac=1).values)
            spam_df[meta_headers] = y.reset_index(drop=True)

            ham_df = df[df['Label'] == 0].reset_index(drop=True)
            y = ham_df[meta_headers].apply(lambda m: m.sample(frac=1).values)
            ham_df[meta_headers] = y.reset_index(drop=True)

        return pd.concat([spam_df, ham_df], axis=0).reset_index(drop=True)

    @staticmethod
    def smote_for_classification(x, y):
        # values to evaluate
        k_values = [1, 2, 3, 4, 5, 6, 7]
        for k in k_values:
            # define pipeline
            model = DecisionTreeClassifier()
            over = SMOTE(sampling_strategy=0.1, k_neighbors=k)
            under = RandomUnderSampler(sampling_strategy=0.5)
            steps = [('over', over), ('under', under), ('model', model)]
            pipeline = Pipeline(steps=steps)

            # evaluate pipeline
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            scores = cross_val_score(pipeline, x, y, scoring='roc_auc', cv=cv, n_jobs=-1)

            return mean(scores)

    @staticmethod
    def borderline_smote(x, y):
        # summarize class distribution
        counter = Counter(y)
        print(counter)

        # transform the dataset
        oversample = BorderlineSMOTE()
        x, y = oversample.fit_resample(x, y)

        # summarize the new class distribution
        return Counter(y)

    @staticmethod
    def adaptive_synthetic_sampling(x, y):
        # summarize class distribution
        counter = Counter(y)
        print(counter)

        # transform the dataset
        oversample = ADASYN()
        x, y = oversample.fit_resample(x, y)

        # summarize the new class distribution
        return Counter(y)


class DataAugmentationManager:

    def __init__(self):
        self.sdam = NumericalDataAugmentationManager()
        self.teda = TweetEDA()

    def augment_data(self, entry_data=pd.DataFrame(), score_new_objects=False, keep_ua_data=True, total_aug_to_make=100,
                     general_alpha=0, alpha_sr=0.6, alpha_ri=0.1, alpha_rs=0.2, alpha_rd=0.1, increment_alpha=0.05,
                     random_text_aug=False, random_total=0.5,
                     to_file='', from_file=''):

        df = self.teda.wrapper(entry_data, score_new_objects, keep_ua_data, total_aug_to_make, general_alpha, alpha_sr,
                               alpha_ri, alpha_rs, alpha_rd, increment_alpha, random_text_aug, random_total, '',
                               from_file)

        meta_headers = df.columns.tolist()
        meta_headers = meta_headers[7:-2]

        df = self.sdam.shuffle_meta_scores(df, meta_headers, group=False)

        if to_file:
            try:
                df.to_csv(to_file, index=False)
            except Exception as e:
                print(e)
                print(f'File {to_file} not found, dataframe not written')

        return df


def main():
    dam = DataAugmentationManager()
    dam.augment_data(random_text_aug=True, random_total=0.6, total_aug_to_make=1000,
                     to_file='../data/Learning Data/augmented_spam_learning.csv',
                     from_file='../data/Learning Data/spam_learning.csv')


if __name__ == '__main__':
    main()
