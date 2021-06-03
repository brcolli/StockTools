import importlib
from os import path


Utils = importlib.import_module('utilities').Utils
Sqm = importlib.import_module('SqliteManager').SqliteManager
TCM = importlib.import_module('TdaClientManager').TdaClientManager


class OptionsDataManager:

    def __init__(self):

        self.database_path = '../data/Databases/options_data.sqlite'
        #self.database = Sqm(self.database_path)


def main():

    odm = OptionsDataManager()

    tcm = TCM()

    tcm.get_option_historical_data()

    #if not odm.database.database_empty():
    #    odm.create_options_data()


if __name__ == '__main__':
    main()
