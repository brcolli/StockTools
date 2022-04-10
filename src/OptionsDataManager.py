import utilities
import SqliteManager
import TdaClientManager


Utils = utilities.Utils
Sqm = SqliteManager.SqliteManager
TCM = TdaClientManager.TdaClientManager


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
