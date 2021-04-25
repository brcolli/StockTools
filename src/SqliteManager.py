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

    def execute_many_query(self, query, data):

        cursor = self.connection.cursor()
        try:
            cursor.executemany(query, data)
            self.connection.commit()
        except sqlite3.Error as e:
            print(f"The error '{e}' occurred")

    def execute_read_query(self, query):

        cursor = self.connection.cursor()
        try:
            cursor.execute(query)
            return cursor.fetchall()
        except sqlite3.Error as e:
            print(f"The error '{e}' occurred")

    def get_column_names(self, table_name):
        cursor = self.connection.execute('SELECT * from {}'.format(table_name))
        return list(map(lambda x: x[0], cursor.description))

    def add_new_column_with_data(self, table_name, id_name, column_name, data_type, data):

        columns = self.get_column_names(table_name)
        if column_name not in columns:

            cursor = self.connection.cursor()
            cursor.execute('ALTER TABLE {} ADD COLUMN IF NOT EXISTS {} {}'.format(table_name, column_name, data_type))

        q = 'UPDATE {} SET {} = ? WHERE {} = ?'.format(table_name, column_name, id_name)
        return self.execute_many_query(q, data)

    def database_empty(self):

        cursor = self.connection.cursor()
        cursor.execute('SELECT name FROM sqlite_master WHERE type=\'table\';')
        return bool(cursor.fetchall())

    def check_if_key_exists(self, table_name, column_name, key):

        cursor = self.connection.cursor()
        try:
            cursor.execute('SELECT EXISTS(SELECT 1 FROM {} WHERE {}=\'{}\')'.format(table_name, column_name, key))
            return cursor.fetchone()[0]
        except sqlite3.Error as e:
            print(f"The error '{e}' occurred")

    def get_first_row(self, table_name, column_sort):
        cursor = self.connection.execute('SELECT * FROM {} ORDER BY {} ASC LIMIT 1;'.format(table_name, column_sort))
        return cursor.fetchone()

    def get_last_row(self, table_name, column_sort):
        cursor = self.connection.execute('SELECT * FROM {} ORDER BY {} DESC LIMIT 1;'.format(table_name, column_sort))
        return cursor.fetchone()

    # List of all SQLITE reserved keywords
    @staticmethod
    def get_sqlite_keywords():
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
