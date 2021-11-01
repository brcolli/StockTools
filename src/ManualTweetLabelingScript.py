import pandas as pd
import importlib
import webbrowser
from pynput import keyboard
from tkinter import filedialog as fd
from tkinter import Tk

Utils = importlib.import_module('utilities').Utils

class Labeler:
    def __init__(self, source_data_path):
        self.source_data_path = source_data_path
        self.TINPUT = -1
        self.hotkeys = '103rslbp'
        self.col_list = ["Tweet id", "Label"]
        self.tweet_base_url = 'https://twitter.com/i/web/status/'
        self.tweet_csv = pd.read_csv(self.source_data_path)
        # self.tweet_csv.sort_values('Label', inplace=True)
        self.pause = False
        try:
            self.count = list(self.tweet_csv['Label']).index(-2)
        except ValueError:
            print("Everything already labeled")
            quit()
        print(f"Starting at #{self.count}\n")

    def on_press(self, key):
        pass

    def on_release(self, key):
        if key == keyboard.Key.pause:
            self.TINPUT = 'p'
            return False
        else:
            try:
                value = key.char
                if value in self.hotkeys and value != 'p':
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
        print('\n')
        print(f'{self.count}\t{url}')
        # webbrowser.open_new(url)
        webbrowser.open(url)

        self.TINPUT = -1

        while str(self.TINPUT) not in self.hotkeys:
            self.wait_for_label()
            if self.TINPUT == 'p':
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
        except Exception as e:
            self.tweet_csv.to_csv('backup.csv', index=False)


def main(file_path, clear_data=False):
    if not file_path:
        Tk().withdraw()
        file_path = fd.askopenfilename()
    labeler = Labeler(file_path)
    v = labeler.tweet_csv['Label'].value_counts()
    print(f'Labeling: "{file_path}"\nClean: {v[0]} Spam: {v[1]} Skipped: {v[-1]} Unlabeled: {v[-2]} Total: '
          f'{len(labeler.tweet_csv)}')
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
