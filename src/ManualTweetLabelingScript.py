import pandas as pd
import importlib
import webbrowser
from pynput import keyboard
from tkinter import filedialog as fd
from tkinter import Tk
import time

Utils = importlib.import_module('utilities').Utils


class Labeler:
    def __init__(self, source_data_path):
        self.source_data_path = source_data_path
        self.TINPUT = -1
        self.hotkeys = '103rslb'
        self.col_list = ["Tweet id", "Label"]
        self.tweet_base_url = 'https://twitter.com/i/web/status/'
        self.tweet_csv = pd.read_csv(self.source_data_path)
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
        self.count = 0

    def loop(self):
        url = self.tweet_base_url + str(self.tweet_csv['Tweet id'][self.count])

        webbrowser.open_new(url)
        self.TINPUT = -1
        self.wait_for_label()

        print(f'{self.count}\t{url}\n')

        if self.TINPUT == '0':
            self.tweet_csv.at[self.count, 'Label'] = 0
        elif self.TINPUT == '1':
            self.tweet_csv.at[self.count, 'Label'] = 1
        elif self.TINPUT == 'r':
            self.tweet_csv.at[self.count, 'Label'] = -1
        elif self.TINPUT == 'b':
            self.count -= 1
            self.tweet_csv.at[self.count, 'Label'] = -2
            self.count -= 1
        elif self.TINPUT == '3':
            self.tweet_csv.at[self.count, 'Label'] = 2
        elif self.TINPUT == 's':
            return [False, True]
        elif self.TINPUT == 'l':
            return [False, False]
        else:
            return [False, True]

        self.count += 1

        if self.count >= len(self.tweet_csv):
            return [False, True]

        else:
            return [True, False]

    def save(self):
        try:
            self.tweet_csv.to_csv(self.source_data_path, index=False)
        except Exception as e:
            self.tweet_csv.to_csv('backup.csv', index=False)


def main(file_path, clear_data=False):
    if not file_path:
        Tk().withdraw()
        file_path = fd.askopenfilename()
    labeler = Labeler(file_path)
    if clear_data:
        labeler.clear_labels()
    active = True
    save = False
    while active:
        active, save = labeler.loop()

    if save:
        labeler.save()


if __name__ == '__main__':
    main('')
