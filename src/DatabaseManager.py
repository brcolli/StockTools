import importlib
import string
import json
from os import path
from os import listdir
import binascii


Utils = importlib.import_module('utilities').Utils
Sqm = importlib.import_module('SqliteManager').SqliteManager


class DatabaseManager:

    def __init__(self):

        self.fundamentals = Sqm('../data/yahoo_stats.sqlite')
        self.nasdaq_shorts = Sqm('../data/nasdaq_shorts.sqlite')
        self.shorts = Sqm('../data/ychart_shorts.sqlite')

    @staticmethod
    def get_data_from_amibroker():

        amibroker_data = 'C:/Program Files/AmiBroker/Data/'

        for letter in string.ascii_lowercase:

            curr_dir = amibroker_data + letter + '/'
            for filename in listdir(curr_dir):

                with open(curr_dir + filename, 'rb') as f:

                    data = f.readlines()
                    for line in data:
                        print(line.decode('ansi'))

            break


def main():

    dbm = DatabaseManager()
    dbm.get_data_from_amibroker()


if __name__ == '__main__':
    main()
