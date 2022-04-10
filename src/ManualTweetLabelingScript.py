import pandas as pd
import webbrowser
from pynput import keyboard
from tkinter import filedialog as fd
from tkinter import Tk
import TwitterSpamModel
import NLPSentimentCalculations
import tweepy
import TwitterManager
import BotometerRequests


SpamModelParameters = TwitterSpamModel.SpamModelParameters
SpamModelData = TwitterSpamModel.SpamModelData
SpamModelLearning = TwitterSpamModel.SpamModelLearning
NSC = NLPSentimentCalculations.NLPSentimentCalculations
TwitterManager = TwitterManager.TwitterManager
BotometerRequests = BotometerRequests.BotometerRequests


class Labeler:

    def __init__(self, source_data_path):

        self.source_data_path = source_data_path
        self.TINPUT = -1
        self.hotkeys = '103rslb`'
        self.col_list = ["Tweet id", "Label"]
        self.tweet_base_url = 'https://twitter.com/i/web/status/'
        self.tweet_csv = pd.read_csv(self.source_data_path)
        self.spam_model_learning = None
        # self.tweet_csv.sort_values('Label', inplace=True)
        self.pause = False
        try:
            self.count = list(self.tweet_csv['Label']).index(-2)
        except ValueError:
            print("Everything already labeled")
            quit()
        print(f"Starting at #{self.count}\n")
        self.tm = TwitterManager()
        self.br = BotometerRequests()

        self.features_to_train = ['full_text', 'cap.english', 'cap.universal',
                                  'raw_scores.english.overall',
                                  'raw_scores.universal.overall',
                                  'raw_scores.english.astroturf',
                                  'raw_scores.english.fake_follower',
                                  'raw_scores.english.financial',
                                  'raw_scores.english.other',
                                  'raw_scores.english.self_declared',
                                  'raw_scores.english.spammer',
                                  'raw_scores.universal.astroturf',
                                  'raw_scores.universal.fake_follower',
                                  'raw_scores.universal.financial',
                                  'raw_scores.universal.other',
                                  'raw_scores.universal.self_declared',
                                  'raw_scores.universal.spammer',
                                  'favorite_count', 'retweet_count']

    def on_press(self, key):
        pass

    def on_release(self, key):
        if key == keyboard.Key.pause:
            self.TINPUT = '`'
            return False
        else:
            try:
                value = key.char
                if value in self.hotkeys:
                    self.TINPUT = value
                    return False
            except ValueError as _:
                key = -1

        return key

    def wait_for_label(self):
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()

    def clear_labels(self):
        self.tweet_csv['Label'] = -2
        print("All labels cleared to -2\n")
        self.count = 0

    def loop(self):

        tid = self.tweet_csv['Tweet id'][self.count]

        tobj = self.tm.twitter_api.get_status(tid, tweet_mode='extended')
        url = self.tweet_base_url + str(tid)

        tjson = tobj._json
        uid = tobj.user.id

        tdf = self.br.wrapper(bot_user_request=True, tweet_objects=[tjson], user_ids=[uid],
                              tweet_ids=[tid], existing_api=self.tm.twitter_api,
                              wanted_bot_keys=self.features_to_train[1:17])

        # Add full_text, retweets, and favorites to df
        tdf['full_text'] = tobj.full_text
        tdf['retweet_count'] = tobj.retweet_count
        tdf['favorite_count'] = tobj.favorite_count

        spam_prediction = self.spam_model_learning.predict(tweet_df=tdf)
        self.tweet_csv.at[self.count, 'Spam Label'] = spam_prediction[0]

        print('\n')
        print(f'{self.count}\t{url}')
        webbrowser.open(url)

        self.TINPUT = -1

        while str(self.TINPUT) not in self.hotkeys:
            self.wait_for_label()
            if self.TINPUT == '`':
                self.pause = not self.pause
                self.TINPUT = -1
                if self.pause:
                    print('\nPaused\n')
                else:
                    print('\nUnpaused\n')
            elif self.pause:
                self.TINPUT = -1
            elif self.TINPUT == '0':
                self.tweet_csv.at[self.count, 'Label'] = 0
                print('Labeled Clean')
                self.count += 1
            elif self.TINPUT == '1':
                self.tweet_csv.at[self.count, 'Label'] = 1
                print('Labeled Spam')
                self.count += 1
            elif self.TINPUT == 'r':
                self.tweet_csv.at[self.count, 'Label'] = -1
                print('Label Skipped')
                self.count += 1
            elif self.TINPUT == 'b':
                self.count -= 1
                self.tweet_csv.at[self.count, 'Label'] = -2
                print('Back to previous Tweet')
            elif self.TINPUT == '3':
                self.tweet_csv.at[self.count, 'Label'] = 2
                print('Labeled Special Label')
                self.count += 1
            elif self.TINPUT == 's':
                print('Saving all changes')
                return [False, True]
            elif self.TINPUT == 'l':
                print('Discarding all changes')
                return [False, False]

        if self.count >= len(self.tweet_csv):
            print('All Tweets labeled, saving')
            return [False, True]

        else:
            return [True, False]

    def save(self):
        try:
            self.tweet_csv.to_csv(self.source_data_path, index=False)
        except Exception as _:
            self.tweet_csv.to_csv('backup.csv', index=False)


def main(file_path, label_spam=False, clear_data=False):

    if not file_path:
        Tk().withdraw()
        file_path = fd.askopenfilename()

    if not file_path:
        return

    labeler = Labeler(file_path)
    v = labeler.tweet_csv['Label'].value_counts()
    labels = {0: 'Clean', 1: 'Spam', -1: 'Skipped', -2: 'Unlabeled'}

    print(f'Labeling: "{file_path}"\n')

    for k in v.keys():
        print(f'{labels[k]}: {v[k]}', end='\t')
    print(f'Total: {len(labeler.tweet_csv)}')

    if clear_data:
        labeler.clear_labels()

    if label_spam:

        labeler.tweet_csv['Spam Label'] = [-2]*len(labeler.tweet_csv['Label'])

        spam_model_params = SpamModelParameters(epochs=1000,
                                                batch_size=128,
                                                load_model=True,
                                                checkpoint_model=False,
                                                saved_model_bin='../data/analysis/Model Results/Saved Models/'
                                                                'best_spam_model.h5')

        spam_model_data = SpamModelData(nsc=NSC(), base_data_csv='../data/Learning Data/spam_learning.csv',
                                        test_size=0.01,
                                        features_to_train=labeler.features_to_train)

        labeler.spam_model_learning = SpamModelLearning(spam_model_params, spam_model_data)
        labeler.spam_model_learning.build_model()

    active = True
    save = False
    while active:
        active, save = labeler.loop()

    if save:
        labeler.save()


if __name__ == '__main__':
    main('../data/Learning Data/stocks_to_label_ben.csv', label_spam=True)
