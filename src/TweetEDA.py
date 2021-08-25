import nltk
import eda
import pandas as pd
import BotometerRequests
import utilities
nltk.download('wordnet')

Utils = utilities.Utils()


class TweetEDA:
    """
    Used to augment tweets using eda.py augmentation techniques, also has remove url method
    """
    def __init__(self):
        pass

    @staticmethod
    def safe_str_to_dict(string: str or dict) -> dict:
        """
        If you are unsure whether you are working with a dictionary or string representation of dictionary, returns a
        dictionary

        :param string: Either string (json) dictionary or dictionary (generally a json of a stored tweet)
        :type string: str or dict

        :return: The dictionary version of the string (or dictionary itself)
        :rtype: dict
        """
        if type(string) == str:
            string = eval(string)
        return string

    def remove_urls(self, json: str or dict) -> str:
        """
        A useful method to clean urls from Tweet text. Takes a tweet object and uses different information to remove
        urls. Then returns the tweet text without urls.

        :param json: Tweet object json, either as a str or dict
        :type json: str or dict

        :return: The text of the Tweet without any urls in it
        :rtype: str
        """

        json = self.safe_str_to_dict(json)
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
    def augment_sentences(sentences: [str], alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, alpha_rd=0.1, num_aug=9) -> [str]:
        """
        Augments sentences using eda.py

        :param sentences: Tweet texts to augment
        :type sentences: [str]

        Read about alpha_sr, alpha_ri, alpha_rs, alpha_rd in the paper or GitHub page

        :param num_aug: Number of augmented strings to make for each original
        :type num_aug: int

        :return: List of augmented strings segmented into lists by original string
                Ex: sentences = ["abc", "def"] num_aug = 3
                return: [["abc1", "abc2", "abc3"], ["def1", "def2", "def3"]]
        :rtype: [[str]]
        """
        augmented_sentences = [eda.eda(s, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd,
                                       num_aug=num_aug)[:-1] for s in sentences]
        return augmented_sentences

    def text_from_tweet(self, string_json: str or dict) -> str:
        """
        Extracts the Tweet text from a str or dictionary Tweet json objects. Necessary because some Tweets have the key
        'full_text', while others have 'text'.

        :param string_json: Tweet object json, either as a str or dict
        :type string_json: str or dict

        :return: Text of the Tweet
        :rtype: str
        """
        string_json = self.safe_str_to_dict(string_json)
        try:
            txt = string_json['full_text']
        except KeyError:
            try:
                txt = string_json['text']
            except KeyError:
                txt = ''

        return txt

    def insert_text_into_json(self, string_json: str or dict, text: str) -> dict:
        """
        Replaces the existing text of a Tweet json object with a new text.

        :param string_json: Tweet object json, either as a str or dict
        :type string_json: str or dict
        :param text: Text to insert instead of the current text
        :type text: str

        :return: Tweet object json, as a dict
        :rtype: dict
        """
        dictionary = self.safe_str_to_dict(string_json)
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
        return [self.insert_text_into_json(o, self.remove_urls(o)) for o in objects]

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

        :return: Dataframe with keys ['Tweet id', 'text', 'json', 'botscore']
        :rtype: pd.DataFrame
        """
        br = BotometerRequests.BotometerRequests()
        scored = br.wrapper(from_dataframe=df, lite_tweet_request=True)
        df['botscore'] = scored['botscore']
        df = Utils.order_dataframe_columns(df, ['Tweet id', 'text', 'json', 'botscore'])
        return df

    def wrapper(self, entry_data: pd.DataFrame, score_new_objects=False) -> pd.DataFrame:
        """
        A method to get EDA augmented TweetObjects and their botometer lite scores

        :param entry_data: The baseline data formatted as a df. Required keys: ['Tweet id', 'json']
        :type entry_data: pd.DataFrame

        :param score_new_objects: Whether or not to score the newly augmented Tweets with Botometer Lite
        :type score_new_objects: bool

        :return: Dataframe with the keys: ['Tweet id', 'text', 'json', 'botscore'(optional)]. Tweet ids for augmented
                tweets are generated as (Original_id * 100) + n. Text has the augmented text of the tweet. json has the
                augmented text reinserted into the Tweet objects. Defaults for augmentation configured in
                self.augment_sentences. Storing each augmented tweet reinserted into the original tweet object takes
                significantly more space than storing only augmented text.
        :rtype: pd.DataFrame
        """

        ids = entry_data['Tweet id'].tolist()
        tweet_objects = [self.insert_text_into_json(s, self.remove_urls(s)) for s in entry_data['json']]
        sentences = [self.text_from_tweet(t) for t in tweet_objects]
        aug_s = self.augment_sentences(sentences)

        aps = len(aug_s[0])

        new_ids = Utils.flatten([list(range((id*100)+1, (id*100)+aps+1)) for id in ids])
        new_objects = [self.insert_text_into_json(tweet_objects[i], aug_s[i][j]) for i in range(len(tweet_objects)) for
                       j in range(aps)]
        new_sentences = Utils.flatten(aug_s)

        df = pd.DataFrame({'Tweet id': new_ids, 'text': new_sentences, 'json': new_objects})
        if score_new_objects:
            df = self.score_new_objects(df)

        return df
