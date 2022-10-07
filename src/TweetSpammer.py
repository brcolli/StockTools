import os
import time
from datetime import datetime
import pandas as pd
import TweetCollector as TC
from TweetDatabaseManager import TweetDatabaseManager as TM

RUN_TIME_HRS = 0.25
HRS_TO_SEC = 3600
START_TIME = time.time()
END_TIME = START_TIME + (RUN_TIME_HRS * HRS_TO_SEC)

TRACKER_DB_PATH = 'SNPTrackerDB.csv'
LOG_PATH = 'log.txt'


def get_datetime_str():
    return str(datetime.now())


def log(msg):
    with open(LOG_PATH, 'a') as f:
        t = f'{get_datetime_str()}\t{msg}\n'
        print(t)
        f.write(t)


def log_space():
    with open(LOG_PATH, 'a') as f:
        t = '\n------------------------\n\n'
        print(t)
        f.write(t)


class TweetDBCollector:
    def __init__(self):
        self.db_df = pd.read_csv(TRACKER_DB_PATH)
        self.tm = TM()
        self.min_rq_size = 100

    def calc_next_rq_amount(self, prev_rq, num_new_tweets):
        # Calculates "optimal" next request size based on the amount of Tweets that were new (not duplicates in old rq)
        percent_new = num_new_tweets / prev_rq
        if percent_new > 0.9:
            next_rq = 2 * prev_rq
        elif percent_new >= 0.5:
            next_rq = percent_new * prev_rq
        else:
            next_rq = max(self.min_rq_size, prev_rq / 2)

        return int(next_rq)

    def request_tweets(self, term, num):
        df = None
        for i in range(10):
            try:
                df = TC.collect_tweets(phrase=term, history_count=num)
                break

            except Exception as e:
                print(e.__str__())
                log(f'Exception: {e.__str__()} ... Sleeping for {10} seconds')
                time.sleep(i*10)

        return df


    def collect_for_term(self, row):
        name, ticker, s1, s2, total_tweets, col_time, num_att, goal, rq_size =\
            self.db_df.loc[row, :].values.flatten().tolist()

        log(f'{name} attempting to collect... Search terms: {s1}, {s2}; Current total: {total_tweets}; '
            f'Collection time: {col_time}; Attempt number: {num_att+1}; Goal: {goal}; Request size: {rq_size}')

        if total_tweets >= goal:
            log(f'{name} goal of {goal} reached, skipping...')

        else:
            start_time = time.time()
            s1_df = self.request_tweets(s1, rq_size)
            s2_df = self.request_tweets(s2, rq_size)

            if s1_df is None and s2_df is None:
                df = None
            elif s1_df is None:
                df = s2_df
            elif s2_df is None:
                df = s1_df
            else:
                df = self.tm.merge_no_duplicates([s1_df, s2_df])

            if df is None:
                new_num = 0
            else:
                if len(df) == 0:
                    new_num = 0
                else:
                    if os.path.exists(f'{name}.csv'):
                        old_df = pd.read_csv(f'{name}.csv')
                        new_df = self.tm.merge_no_duplicates([old_df, df])
                        new_num = len(new_df) - len(old_df)
                        if new_num > 0:
                            new_df.to_csv(f'{name}.csv', index=False)
                    else:
                        new_num = len(df)
                        df.to_csv(f'{name}.csv', index=False)

            total_tweets = total_tweets + new_num
            col_time = col_time + (time.time() - start_time)
            num_att += 1
            rq_size = self.calc_next_rq_amount(rq_size, new_num)

            log(f'{name} collection stats... New total: {total_tweets}; New: {new_num}; Collection time: {col_time}; '
                f'New request size: {rq_size}')

            self.db_df.loc[row] = [name, ticker, s1, s2, total_tweets, col_time, num_att, goal, rq_size]
            self.db_df.to_csv(TRACKER_DB_PATH, index=False)


    def run(self):
        log('Starting collection')
        row_n = 0
        while time.time() < END_TIME:
            log_space()
            self.collect_for_term(row_n)
            row_n += 1
            if row_n > len(self.db_df):
                row_n = 0

        log_space()
        log('Ending collection')




# count = 0
# df1 = pd.DataFrame(columns=['Tweet id', 'User id', 'Screen name', 'Label', 'Search term', 'json'])
# old_df_size = 0
# prev_path = ''
#
#
# search_term = 'InterestRates'
# phrase = 'Interest Rates'
#
# dir_path = f'../data/TweetData/{search_term}/'
# while True:
#     if count == 0:
#         rq = 50000
#     else:
#         rq = 100
#     try:
#         df = TC.collect_tweets(phrase=phrase, history_count=rq)
#
#     except Exception as e:
#         print(e.__str__())
#         msg = f'Iteration {count} ... Exception: {e.__str__()} ... Sleeping for {10} seconds'
#         with open(f'{dir_path}log.txt', 'a') as f:
#             f.write(msg + '\n')
#
#         time.sleep(10)
#         continue
#
#     df1 = TM.merge_no_duplicates([df1, df])
#     new_num = len(df1) - old_df_size
#
#     if new_num > 0:
#         df1.to_csv(f'{dir_path}{search_term}622-{count}.csv', index=False)
#         if prev_path:
#             os.remove(prev_path)
#         prev_path = f'{dir_path}{search_term}622-{count}.csv'
#
#     sleep = max(160 - new_num, 0)
#
#     msg = f'Iteration {count} ... Collected {len(df)} Tweets ... {new_num} are new ... Sleeping for {sleep} seconds'
#     print('---------')
#     print(msg)
#     print('---------')
#     with open(f'{dir_path}log.txt', 'a') as f:
#         f.write(msg+'\n')
#
#     df = pd.DataFrame()
#     old_df_size = len(df1)
#     count += 1
#     time.sleep(sleep)
#
