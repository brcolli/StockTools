import pandas as pd
import importlib
import webbrowser
from pynput import keyboard
import time

Utils = importlib.import_module('utilities').Utils


class Labeler:
    def __init__(self, source_data_path, output_path, output_file_name):
        self.source_data_path = source_data_path
        self.output_path = output_path
        self.output_file_name = output_file_name
        self.TINPUT = -1
        self.hotkeys = 'aqrsl'
        self.col_list = ["Tweet id", "Label"]
        self.tweet_base_url = 'https://twitter.com/i/web/status/'
        self.tweet_csv = pd.read_csv(self.source_data_path, usecols=self.col_list, dtype={'Tweet id': str, 'Label': int})
        try:
            self.count = list(self.tweet_csv['Label']).index(-2)
        except ValueError:
            self.count = 0
        print(f"Starting at #{self.count}\n")

    def on_press(self, key):
        pass

    def on_release(self, key):
        print(key)
        try:
            value = key.char
            if value in self.hotkeys:
                self.TINPUT = value
                return False
        except:
            key = -1
        return key

    def wait_for_label(self):
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()

    def clear_labels(self):
        self.tweet_csv['Label'] = -2
        print("All labels cleared to -2\n")

    def loop(self):
        url = self.tweet_base_url + self.tweet_csv['Tweet id'][self.count]

        webbrowser.open_new(url)
        self.TINPUT = -1
        self.wait_for_label()

        print(f'{self.count}\t{url}\n')

        if self.TINPUT == 'a':
            self.tweet_csv.at[self.count, 'Label'] = 0
        elif self.TINPUT == 'q':
            self.tweet_csv.at[self.count, 'Label'] = 1
        elif self.TINPUT == 'r':
            self.tweet_csv.at[self.count, 'Label'] = -1
        elif self.TINPUT == 's':
            return [False, True]
        elif self.TINPUT == 'l':
            return [False, False]
        else:
            return [False, True]

        self.count += 1
        return [True, False]

    def save(self):
        if not Utils.write_dataframe_to_csv(self.tweet_csv, self.output_path+self.output_file_name, write_index=False):
            Utils.write_dataframe_to_csv(self.tweet_csv, f'{self.output_path}{time.time()}.csv', write_index=False)


def main(clear_data, source_data_path, output_path, output_file):
    labeler = Labeler(source_data_path, output_path, output_file)
    if clear_data:
        labeler.clear_labels()
    active = True
    save = False
    while active:
        active, save = labeler.loop()

    if save:
        labeler.save()


if __name__ == '__main__':
    main(False, '../data/BotometerTesting2/Stock_tweet_history_search.csv', '../data/BotometerTesting2/', 'Stock200Labeled.csv')
