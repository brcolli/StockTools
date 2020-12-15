import sqlite3


class SqliteManager:

    def __init__(self, path='../data/data.sqlite'):
        self.connection = None
        try:
            self.connection = sqlite3.connect(path)
        except sqlite3.Error as e:
            print(f"The error '{e}' occurred")

    def execute_query(self, query):

        cursor = self.connection.cursor()
        try:
            cursor.execute(query)
            self.connection.commit()
        except sqlite3.Error as e:
            print(f"The error '{e}' occurred")

    def execute_read_query(self, query):

        cursor = self.connection.cursor()
        try:
            cursor.execute(query)
            result = cursor.fetchall()
            return result
        except sqlite3.Error as e:
            print(f"The error '{e}' occurred")