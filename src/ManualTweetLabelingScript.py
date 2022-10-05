import pandas as pd
import webbrowser
from pynput import keyboard
from tkinter import filedialog as fd
from tkinter import Tk
import time


class Labeler:

    def __init__(self, source_data_path: str, label_names: [str], label_keys: [str], label_values: [int], label_col):

        self.lc = label_col

        # Load Tweet csv from file and sort by labels
        self.source_data_path = source_data_path
        self.tweet_csv = pd.read_csv(self.source_data_path)

        if self.lc not in self.tweet_csv.columns:
            self.tweet_csv[self.lc] = -2

        # Check if there are Tweets left to label
        try:
            self.count = list(self.tweet_csv[self.lc]).index(-2)
        except ValueError:
            print("Everything already labeled")
            quit()

        self.TINPUT = -1
        self.action_names = ['Save', 'Discard', 'Go Back to Previous Tweet', 'Pause'] + label_names
        self.action_values = [None, None, -2, None] + label_values
        self.hotkeys = 'slb`' + ''.join(label_keys)
        self.pause = False

        self.tweet_base_url = 'https://twitter.com/i/web/status/'

        print(f"Starting at #{self.count}\n")

    # Do nothing on press
    def on_press(self, key):
        pass

    # Get key pressed on release
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
            except AttributeError as _:
                key = -1

        return key

    def wait_for_label(self):
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()

    def clear_labels(self, clear_val: int):
        self.tweet_csv[self.lc] = clear_val
        print(f"All labels cleared to {clear_val}\n")
        self.count = 0

    def print_help(self):
        print("Keys and their actions: \n")
        for i in range(len(self.action_names)):
            if i <= 3:
                print(f'Hit {self.hotkeys[i]} to {self.action_names[i]}')
            else:
                print(f'Hit {self.hotkeys[i]} to label as {self.action_names[i]}')

    def loop(self):

        url = self.tweet_base_url + str(self.tweet_csv['Tweet id'][self.count])

        print('\n')
        print(f'{self.count}\t{url}')

        webbrowser.open(url)

        # User input variable
        self.TINPUT = -1

        while str(self.TINPUT) not in self.hotkeys:
            # Wait for keyboard press
            self.wait_for_label()

            key_index = self.hotkeys.index(str(self.TINPUT))

            # Default action like Pause, Save, Discard, Back
            if key_index <= 3:
                # Pause Toggle
                if self.TINPUT == '`':
                    self.pause = not self.pause
                    self.TINPUT = -1
                    if self.pause:
                        print('\nPaused\n')
                    else:
                        print('\nUnpaused\n')

                # Pause mode (ignore)
                elif self.pause:
                    self.TINPUT = -1

                # Save
                elif self.TINPUT == 's':
                    print('Saving all changes')
                    return [False, True]

                # Discard changes
                elif self.TINPUT == 'l':
                    print('Discarding all changes')
                    return [False, False]

                # Back
                elif self.TINPUT == 'b':
                    self.count -= 1
                    self.tweet_csv.at[self.count, self.lc] = self.action_values[key_index]
                    print('Back to previous Tweet')

            # Label given
            else:
                if self.pause:
                    self.TINPUT = -1
                else:
                    label = self.action_names[key_index]
                    lv = self.action_values[key_index]

                    print(f'Labeled as {lv}: {label}')
                    self.tweet_csv.at[self.count, self.lc] = lv
                    self.count += 1

        # Out of Tweets to label
        if self.count >= len(self.tweet_csv):
            print('\nAll Tweets labeled, saving')
            return [False, True]

        # Continue loop normally
        else:
            return [True, False]

    def save(self):
        try:
            self.tweet_csv.to_csv(self.source_data_path, index=False)
        # Avoid data loss if error
        except Exception as _:
            self.tweet_csv.to_csv(f'{time.time()}-backup.csv', index=False)


def main(file_path, clear_data=False):

    # Use Tkinter to get filepath if not provided
    if not file_path:
        Tk().withdraw()
        file_path = fd.askopenfilename()

    if not file_path:
        return

    # Get Labeling Mode either from File Name or User Input
    mode = ''
    file_name = file_path.split('/')[-1]
    prompt = False

    if 'spam' in file_name and 'sentiment' in file_name:
        prompt = True

    elif 'spam' in file_path:
        mode = 'Spam'

    elif 'sentiment' in file_path:
        mode = 'Sentiment'

    else:
        prompt = True

    if prompt:
        print("\nWould you like to label Spam or Sentiment? \n Enter:\n 0 for Spam \n 1 for Sentiment")
        while not mode:
            x = input("\n")
            try:
                x = int(x)
            except ValueError:
                print("\n Please input: \n 0 for Spam or \n 1 for Sentiment")
                continue

            if x == 0:
                mode = 'Spam'

            elif x == 1:
                mode = 'Sentiment'

    # Set column name according to naming convention
    label_column = mode + 'ManualLabel'

    labels = [0, 1, 2, -1, -2]
    keys = ['q', 'w', 'e', 'r', 't']
    names = []
    if mode == 'Spam':
        names = ['Clean', 'Spam', 'Middle']
    elif mode == 'Sentiment':
        names = ['Positive', 'Neutral', 'Negative']
    names = names + ['Skipped', 'Special']

    # Load File, clear if needed
    labeler = Labeler(file_path, names, keys, labels, label_column)
    if clear_data:
        labeler.clear_labels(-2)

    curr_counts = labeler.tweet_csv[label_column].value_counts()
    print(f'{mode} Labeling: "{file_path}"\n')

    print_dict = dict(zip(labels, names))
    print_dict[-2] = 'Unlabeled'
    for k in curr_counts.keys():
        print(f'{print_dict[k]}: {curr_counts[k]}', end='\t')
    print(f'Total: {len(labeler.tweet_csv)}\n')

    labeler.print_help()
    print()

    # Run loop
    active = True
    save = False
    while active:
        active, save = labeler.loop()

    if save:
        labeler.save()


if __name__ == '__main__':
    main('../data/TweetData/Tweets/BenSentiment416.csv')
