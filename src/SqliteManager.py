import sqlite3


"""SqliteManager

Description:
Module for handling generic Sqlite3 database calls. Used for all sqlite3 database backend API methods.

Authors: Benjamin Collins
Date: April 22, 2021 
"""


class SqliteManager:

    """Class for all generic Sqlite3 backend methods.
    """

    def __init__(self, path='../data/data.sqlite'):

        """Constructor method, opens a connection to a given database path.

        :param path: Path to sqlite database; defaults to ../data/data.sqlite
        :type path: str
        """

        self.connection = None
        try:
            self.connection = sqlite3.connect(path)
        except sqlite3.Error as e:
            print(f"The error '{e}' occurred")

    def execute_query(self, query):

        """Executes a PUT-style query on the database.

        :param query: Sqlite-style database query
        :type query: str
        """

        cursor = self.connection.cursor()
        try:
            cursor.execute(query)
            self.connection.commit()
        except sqlite3.Error as e:
            print(f"The error '{e}' occurred")

    def execute_many_query(self, query, data):

        """Executes a PUT-style query on the database, but allows for multiple data posts in one command.

        :param query: Sqlite-style database query
        :type query: str
        :param data: List of data to be posted to the database
        :type data: list(tuple(object))
        """

        cursor = self.connection.cursor()
        try:
            cursor.executemany(query, data)
            self.connection.commit()
        except sqlite3.Error as e:
            print(f"The error '{e}' occurred")

    def execute_read_query(self, query):

        """Executes a GET-style query on the database, able to return requested data.

        :param query: Sqlite-style database query
        :type query: str

        :return: Data fetch results
        :rtype: list(object)
        """

        cursor = self.connection.cursor()
        try:
            cursor.execute(query)
            return cursor.fetchall()
        except sqlite3.Error as e:
            print(f"The error '{e}' occurred")

    def create_table(self, table_name, key_type_pairs):

        """Creates a table if it doesn't exist.

        :param table_name: Table name in database
        :type table_name: str
        :param key_type_pairs: List pairs for the column key names to their types, where the first entry MUST be the
                               primary key
        :type table_name: list(str, str)
        """

        q = f'CREATE TABLE IF NOT EXISTS[{table_name}] ('

        for pair in range(len(key_type_pairs)):
            if pair == 0:
                q += f'{key_type_pairs[pair][0]} {key_type_pairs[pair][1]} PRIMARY KEY,'
            else:
                q += f'{key_type_pairs[pair][0]} {key_type_pairs[pair][1]},'
        q = q[:-1] + ');'

        self.execute_query(q)

    def insert_row(self, table_name, row_vals):

        """Inserts a single data row into a table.

        :param table_name: Table name in database
        :type table_name: str
        :param row_vals: An iterable row of objects to insert into the table, where order MUST match the column order
        :type row_vals: iterable(object)
        """

        columns = self.get_column_names(table_name)

        q = f'INSERT INTO {table_name} ('
        q += ', '.join(columns) + ') VALUES (' + ', '.join(map(str, row_vals))
        q = q[:-2] + ');'

        self.execute_query(q)

    def insert_many_rows(self, table_name, data):

        """Inserts many rows into a table.

        :param table_name: Table name in database
        :type table_name: str
        :param data: List of data to be posted to the database
        :type data: list(tuple(object))
        """

        columns = self.get_column_names(table_name)

        q = f'INSERT INTO {table_name} ('
        q += ', '.join(columns) + ') VALUES (' + '?, ' * len(columns)
        q = q[:-2] + ');'

        self.execute_many_query(q, data)

    def get_column_names(self, table_name):

        """Gets the column names of a given table

        :param table_name: Table name in database
        :type table_name: str

        :return: Column names of the table
        :rtype: list(str)
        """

        cursor = self.connection.execute('SELECT * from {}'.format(table_name))
        return list(map(lambda x: x[0], cursor.description))

    def add_new_column_with_data(self, table_name, id_name, column_name, data_type, data):

        """Adds a new column with data for each row to a given table

        :param table_name: Table name in database
        :type table_name: str
        :param id_name: Name of the primary key index column
        :type id_name: str
        :param column_name: Name of the column to add
        :type column_name: str
        :param data_type: Type of data in the column to be added, in sqlite3 format
        :type data_type: str
        :param data: List of tuple data to be added
        :type data: list(tuple(object))

        :return: Data from new column
        :rtype: list(object)
        """

        columns = self.get_column_names(table_name)
        if column_name not in columns:

            cursor = self.connection.cursor()
            cursor.execute('ALTER TABLE {} ADD COLUMN IF NOT EXISTS {} {}'.format(table_name, column_name, data_type))

        q = 'UPDATE {} SET {} = ? WHERE {} = ?'.format(table_name, column_name, id_name)
        return self.execute_many_query(q, data)

    def database_empty(self):

        """Returns true if the database is completely empty, including no tables. Otherwise false.

        :return: True if database is empty, else false
        :rtype: bool
        """

        cursor = self.connection.cursor()
        cursor.execute('SELECT name FROM sqlite_master WHERE type=\'table\';')
        return bool(cursor.fetchall())

    def check_if_key_exists(self, table_name, column_name, key):

        """Searches for a key in a column for a specific table.

        :param table_name: Table name in database
        :type table_name: str
        :param column_name: Name of the column to search
        :type column_name: str
        :param key: Key to search for, represented as a string
        :type key: str

        :return: True if table has exact key/column pair, else false
        :rtype: bool
        """

        cursor = self.connection.cursor()
        try:
            cursor.execute('SELECT EXISTS(SELECT 1 FROM {} WHERE {}=\'{}\')'.format(table_name, column_name, key))
            return cursor.fetchone()[0]
        except sqlite3.Error as e:
            print(f"The error '{e}' occurred")

    def get_first_row(self, table_name, column_sort):

        """Gets the first row in a table based on sorting the given column in ascending order.

        :param table_name: Table name in database
        :type table_name: str
        :param column_sort: Column to sort by in ascending order
        :type column_sort: str

        :return: The first row in the table based on the sorted column
        :rtype: list(object)
        """

        cursor = self.connection.execute('SELECT * FROM {} ORDER BY {} ASC LIMIT 1;'.format(table_name, column_sort))
        return cursor.fetchone()

    def get_last_row(self, table_name, column_sort):

        """Gets the last row in a table based on sorting the given column by sorting in descending order and
        getting the first row.

        :param table_name: Table name in database
        :type table_name: str
        :param column_sort: Column to sort by in descending order
        :type column_sort: str

        :return: The last row in the table based on the sorted column
        :rtype: list(object)
        """

        cursor = self.connection.execute('SELECT * FROM {} ORDER BY {} DESC LIMIT 1;'.format(table_name, column_sort))
        return cursor.fetchone()

    # List of all SQLITE reserved keywords
    @staticmethod
    def get_sqlite_keywords():

        """Returns a string of all the keywords that are reserved by sqlite3 that cannot be used as names or other
        variable naming.

        :return: Whitespace-split string of all sqlite3 reserved keywords
        :rtype: str
        """

        return 'ABORT ACTION ADD AFTER ALL ALTER ALWAYS ANALYZE AND AS ASC ATTACH AUTOINCREMENT BEFORE BEGIN BETWEEN' \
               'BY CASCADE CASE CAST CHECK COLLATE COLUMN COMMIT CONFLICT CONSTRAINT CREATE CROSS CURRENT' \
               'CURRENT_DATE CURRENT_TIME CURRENT_TIMESTAMP DATABASE DEFAULT DEFERRABLE DEFERRED DELETE DESC DETACH' \
               'DISTINCT DO DROP EACH ELSE END ESCAPE EXCEPT EXCLUDE EXCLUSIVE EXISTS EXPLAIN FAIL FILTER FIRST' \
               'FOLLOWING FOR FOREIGN FROM FULL GENERATED GLOB GROUP GROUPS HAVING IF IGNORE IMMEDIATE IN INDEX' \
               'INDEXED INITIALLY INNER INSERT INSTEAD INTERSECT INTO IS ISNULL JOIN KEY LAST LEFT LIKE LIMIT MATCH' \
               'NATURAL NO NOT NOTHING NOTNULL NULL NULLS OF OFFSET ON OR ORDER OTHERS OUTER OVER PARTITION PLAN' \
               'PRAGMA PRECEDING PRIMARY QUERY RAISE RANGE RECURSIVE REFERENCES REGEXP REINDEX RELEASE RENAME REPLACE' \
               'RESTRICT RIGHT ROLLBACK ROW ROWS SAVEPOINT SELECT SET TABLE TEMP TEMPORARY THEN TIES TO TRANSACTION' \
               'TRIGGER UNBOUNDED UNION UNIQUE UPDATE USING VACUUM VALUES VIEW VIRTUAL WHEN WHERE WINDOW WITH WITHOUT'
