import ScrapeYahooStats
from utilities import Utils
import pandas as pd
import sys
import os
import random
import matplotlib as plt

os.chdir('../data/SentimentStockCorrelation')

names = pd.read_csv('sp-100.csv')['Name'].tolist()
start_date = '20220801'
end_date = '20220831'

trading_dates= ['2022-08-01', '2022-08-02', '2022-08-03', '2022-08-04', '2022-08-05', '2022-08-08',
         '2022-08-09', '2022-08-10', '2022-08-11', '2022-08-12', '2022-08-15', '2022-08-16',
         '2022-08-17', '2022-08-18', '2022-08-19', '2022-08-22', '2022-08-23', '2022-08-24',
         '2022-08-25', '2022-08-26', '2022-08-29', '2022-08-30']


def populate_price_with_dates():
    for n in names:
        x = pd.read_csv(f'AllPriceData/{n} PriceData.csv')
        x = x[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]
        x.to_csv(f'AllPriceData/{n} PriceData.csv', index=False)


def gen_date_based_dataframe(names):
    for n in names:
        sentiment_data = pd.read_csv(f'Historic SP-100_{start_date}-{end_date}/{n}{start_date}-{end_date}Labeled_Neu.csv')
        sentiment_data['Date'] = sentiment_data['Timestamp'].apply(lambda x: x[:10])

        # Get average sentiment by day with spam free version
        sentiment_by_day = average_sentiment_data_by_day(sentiment_data)
        sentiment_by_day_spam_free = average_sentiment_data_by_day(sentiment_data, spam_free=True)

        # Merge average sentiment into one df
        sentiment_by_day.rename(columns={'SentimentScore': 'SentimentAverage'})
        sentiment_by_day['SentimentAverageSpamFree'] = sentiment_by_day_spam_free['SentimentScore']
        sentiment_by_day = sentiment_by_day.sort_values(by='Date')

        # Get stock data
        stock_data = pd.read_csv(f'{n} PriceData.csv')
        stock_data = stock_data[['Date', 'Close', 'Volume']]
        stock_data = stock_data.sort_values(by='Date')

        trading_dates = stock_data['Date'].tolist()
        all_dates = sentiment_by_day['Date'].tolist()
        # TODO Write a faster algorithm for adding holiday dates into stock price data
        closes = []
        volumes = []
        j = 0
        for i in range(len(all_dates)):
            if all_dates[i] == trading_dates[j]:
                j += 1
                closes[i] = stock_data.loc[j].at['Close']
                volumes[i] = stock_data.loc[j].at['Volume']

            else:
                if i > 0:
                    closes[i] = closes[i-1]
                else:
                    closes[i] = 0

                volumes[i] = 0

        stock_data['Close'] = closes
        stock_data['Volume'] = volumes


        return stock_data




def average_sentiment_data_by_day(df, spam_free=False):
    if spam_free:
        spam_free_df = df.loc[df['SpamScore'] != 1]
        means = spam_free_df.groupby('Date').mean()
    else:
        means = df.groupby('Date').mean()

    df = pd.DataFrame(columns=['Date', 'SentimentScoreNormal'])
    return means[['Date', 'SentimentScoreRaw']]





# def apple_example():
#     a = pd.read_csv('AAPL.csv')
#     b = pd.DataFrame(columns=['Tweet id', 'SentimentScoreRaw', 'Date'])
#
#     a = a[['Date', 'Close']]
#     a['Date'] = pd.to_datetime(a.Date)
#     a = a.iloc[220:]
#
#     x, y, z = gen_random_data(100000)
#     b['Tweet id'] = x
#     b['SentimentScoreRaw'] = y
#     b['Date'] = z
#     b['Date'] = pd.to_datetime(b.Date)
#     b = b.sort_values('Date')
#     means = b.groupby('Date').mean()
#     a['SentimentScore'] = means['SentimentScoreRaw'].tolist()
#
#     return a
#
# x = apple_example()
#
# def graph(x):
#     max_p = max(x['Close'])
#     min_p = min()
#
#

