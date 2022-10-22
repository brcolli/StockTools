import ScrapeYahooStats
import json
from utilities import Utils
import pandas as pd
import sys
import os
import random
import matplotlib as plt

# from NLPSentimentCalculations import calculate_daily_sentiment_score_og, calculate_daily_sentiment_score_og_with_sub, calculate_daily_sentiment_score_sum

os.chdir('../data/SentimentStockCorrelation/SeptemberBeta10')

names = pd.read_csv('beta_companies.csv')['Name'].tolist()
start_date = '20220901'
end_date = '20221001'

# trading_dates= ['2022-08-01', '2022-08-02', '2022-08-03', '2022-08-04', '2022-08-05', '2022-08-08',
#          '2022-08-09', '2022-08-10', '2022-08-11', '2022-08-12', '2022-08-15', '2022-08-16',
#          '2022-08-17', '2022-08-18', '2022-08-19', '2022-08-22', '2022-08-23', '2022-08-24',
#          '2022-08-25', '2022-08-26', '2022-08-29', '2022-08-30']

all_dates = [
    "2022-09-01",
    "2022-09-02",
    "2022-09-03",
    "2022-09-04",
    "2022-09-05",
    "2022-09-06",
    "2022-09-07",
    "2022-09-08",
    "2022-09-09",
    "2022-09-10",
    "2022-09-11",
    "2022-09-12",
    "2022-09-13",
    "2022-09-14",
    "2022-09-15",
    "2022-09-16",
    "2022-09-17",
    "2022-09-18",
    "2022-09-19",
    "2022-09-20",
    "2022-09-21",
    "2022-09-22",
    "2022-09-23",
    "2022-09-24",
    "2022-09-25",
    "2022-09-26",
    "2022-09-27",
    "2022-09-28",
    "2022-09-29",
    "2022-09-30",
]

trading_dates = [
    "2022-09-01",
    "2022-09-02",
    "2022-09-06",
    "2022-09-07",
    "2022-09-08",
    "2022-09-09",
    "2022-09-12",
    "2022-09-13",
    "2022-09-14",
    "2022-09-15",
    "2022-09-16",
    "2022-09-19",
    "2022-09-20",
    "2022-09-21",
    "2022-09-22",
    "2022-09-23",
    "2022-09-26",
    "2022-09-27",
    "2022-09-28",
    "2022-09-29",
    "2022-09-30"
]


def calculate_daily_sentiment_score_og(sent_scores: dict) -> float:
    """Calculates the sentiment score of a day given a dictionary of sums of confidence scores and counts
    for each label.

    :param sent_scores: Dictionary mapping each label to a dictionary of sums of confidence scores and label counts
    :type sent_scores: dict('0': dict('ConfidenceSum': float, 'Count': int),
    '1': dict('ConfidenceSum': float, 'Count': int),
    '2': dict('ConfidenceSum': float, 'Count': int))

    :return: Calculated daily sentiment score
    :rtype: float
    """

    count = sent_scores['0']['Count'] + sent_scores['1']['Count'] + sent_scores['2']['Count']

    # If we have less than 1 tweet, we can not get a sentiment score for that time-period
    if count < 1:
        print(f"Tweets {count}")
        return -1

    if sent_scores['0']['Count'] == sent_scores['2']['Count']:

        # If equal number of positive and negative, make neutral max
        max_key = '1'
    else:

        max_key = '0'
        max_val = 0
        for skey, sval in sent_scores.items():
            if sval['Count'] > max_val:
                max_val = sval['Count']
                max_key = skey

    sent_sum = sent_scores[max_key]['Count']
    if max_key == '0':
        sent_day = Utils.posnorm(sent_sum, 0, count)
    elif max_key == '1':

        if sent_scores['0']['ConfidenceSum'] > sent_scores['2']['ConfidenceSum']:
            # More positive than negative, skew positive
            sent_day = Utils.neunorm(sent_sum - sent_scores['0']['Count'] +
                                     sent_scores['2']['Count'], 0, count, 65, 50)

        elif sent_scores['0']['ConfidenceSum'] < sent_scores['2']['ConfidenceSum']:
            # More negative than positive, skew negative
            sent_day = Utils.neunorm(sent_sum - sent_scores['2']['Count'] +
                                     sent_scores['0']['Count'], 0, count, 35, 50)

        else:
            # Everything is even, set to true neutral
            sent_day = 50.0
    else:
        sent_day = Utils.negnorm(sent_sum, count, 0)

    return sent_day


def calculate_daily_sentiment_score_og_with_sub(sent_scores: dict) -> float:
    """Calculates the sentiment score of a day given a dictionary of sums of confidence scores and counts
    for each label.

    :param sent_scores: Dictionary mapping each label to a dictionary of sums of confidence scores and label counts
    :type sent_scores: dict('0': dict('ConfidenceSum': float, 'Count': int),
    '1': dict('ConfidenceSum': float, 'Count': int),
    '2': dict('ConfidenceSum': float, 'Count': int))

    :return: Calculated daily sentiment score
    :rtype: float
    """

    count = sent_scores['0']['Count'] + sent_scores['1']['Count'] + sent_scores['2']['Count']

    # If we have less than 1 tweet, we can not get a sentiment score for that time-period
    if count < 1:
        print(f"Tweets {count}")
        return -1

    if sent_scores['0']['Count'] == sent_scores['2']['Count']:

        # If equal number of positive and negative, make neutral max
        max_key = '1'
    else:

        max_key = '0'
        max_val = 0
        for skey, sval in sent_scores.items():
            if sval['Count'] > max_val:
                max_val = sval['Count']
                max_key = skey

    sent_sum = sent_scores[max_key]['Count']
    if max_key == '0':
        sent_day = Utils.posnorm(sent_sum - sent_scores['1']['Count'] * (2 / 3) - sent_scores['2']['Count'], 0,
                                 count)
    elif max_key == '1':

        if sent_scores['0']['ConfidenceSum'] > sent_scores['2']['ConfidenceSum']:
            # More positive than negative, skew positive
            sent_day = Utils.neunorm(sent_sum - sent_scores['0']['Count'] +
                                     sent_scores['2']['Count'], 0, count, 65, 50)

        elif sent_scores['0']['ConfidenceSum'] < sent_scores['2']['ConfidenceSum']:
            # More negative than positive, skew negative
            sent_day = Utils.neunorm(sent_sum - sent_scores['2']['Count'] +
                                     sent_scores['0']['Count'], 0, count, 35, 50)

        else:
            # Everything is even, set to true neutral
            sent_day = 50.0
    else:
        sent_day = Utils.negnorm(sent_sum - sent_scores['1']['Count'] * (2 / 3) - sent_scores['0']['Count'], count,
                                 0)

    return sent_day


def calculate_daily_sentiment_score_sum(sent_scores: dict) -> float:
    """Calculates the sentiment score of a day given a dictionary of sums of confidence scores and counts
    for each label.

    :param sent_scores: Dictionary mapping each label to a dictionary of sums of confidence scores and label counts
    :type sent_scores: dict('0': dict('ConfidenceSum': float, 'Count': int),
    '1': dict('ConfidenceSum': float, 'Count': int),
    '2': dict('ConfidenceSum': float, 'Count': int))

    :return: Calculated daily sentiment score
    :rtype: float
    """

    mean_sum = 0
    count = 0
    for skey, sval in sent_scores.items():
        label = int(skey)
        mean_sum += label * sval['Count']
        count += sval['Count']

    # If we have less than 1 tweet, we can not get a sentiment score for that time-period
    if count < 1:
        print(f"Tweets {count}")
        return -1

    # We multiply by 2 as that is the max label
    sent_day = Utils.normalize(mean_sum, count * 2, 0, 0, 100)

    return sent_day


def populate_price_with_dates():
    for n in names:
        x = pd.read_csv(f'AllPriceData/{n} PriceData.csv')
        print(n, len(x), len(trading_dates))
        x['Date'] = trading_dates
        x = x[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]
        x.to_csv(f'AllPriceData/{n} PriceData.csv', index=False)


def fill_non_trading_days_in_price_data(price_data, all_dates):
    # TODO Write a faster algorithm for adding holiday dates into stock price data

    # For Non-Trading Days
    # Close = Previous trading day close
    # Volume = 0
    closes = [0] * len(all_dates)
    volumes = [0] * len(all_dates)
    j = 0
    for i in range(len(all_dates)):
        if all_dates[i] == price_data.loc[j].at['Date']:
            closes[i] = price_data.loc[j].at['Close']
            volumes[i] = price_data.loc[j].at['Volume']
            j += 1

        else:
            if i > 0:
                closes[i] = closes[i - 1]
            else:
                closes[i] = 0

            volumes[i] = 0

    output_df = pd.DataFrame(columns=['Date', 'Close', 'Volume'])
    output_df['Date'] = all_dates
    output_df['Close'] = closes
    output_df['Volume'] = volumes

    return output_df


def gen_date_based_dataframe(name, all_dates):
    # Load sentiment data
    sentiment_data = pd.read_csv(f'{name}{start_date}-{end_date}Labeled.csv')
    sentiment_data['Date'] = sentiment_data['Timestamp'].apply(lambda x: x[:10])

    # Get daily sentiment normal and spam-free version
    sentiment_by_day = average_sentiment_data_by_day(sentiment_data, all_dates)
    sentiment_by_day_spam_free = average_sentiment_data_by_day(sentiment_data, all_dates, spam_free=True)

    # Merge daily sentiment normal and spam-free version
    sentiment_by_day_spam_free = sentiment_by_day_spam_free.drop('Date', axis=1)
    sentiment_by_day = pd.concat([sentiment_by_day, sentiment_by_day_spam_free], axis=1)
    sentiment_by_day = sentiment_by_day.sort_values(by='Date')

    # Load stock price data
    price_data = pd.read_csv(f'AllPriceData/{name} PriceData.csv')
    price_data = price_data[['Date', 'Close', 'Volume']]
    price_data = price_data.sort_values(by='Date')

    # Fill in stock price for non-trading days
    filled_price_data = fill_non_trading_days_in_price_data(price_data, all_dates)

    # Merge dfs
    filled_price_data = filled_price_data.drop('Date', axis=1)
    output_df = pd.concat([sentiment_by_day, filled_price_data], axis=1)

    return output_df


def gen_all_date_based_dataframes():
    return [gen_date_based_dataframe(n, all_dates) for n in names]


def average_sentiment_data_by_day(df, all_dates, spam_free=False):
    # Different Methods of Getting Daily Average
    ss_average_method_names = ['Og', 'Sum', 'OgSub']
    ss_average_methods = [calculate_daily_sentiment_score_og, calculate_daily_sentiment_score_sum,
                          calculate_daily_sentiment_score_og_with_sub]
    ss_scores = [[], [], []]
    num_tweets = []
    num_tweets_key = "#Tweets"

    if spam_free:
        df = df.loc[df['SpamLabel'] != 1]
        ss_average_method_names = [s + 'SpamFree' for s in ss_average_method_names]
        num_tweets_key = num_tweets_key + 'SpamFree'

    df = df[['Date', 'SentimentLabel', 'SentimentConfidence']]
    split_by_date_and_label = df.groupby(['Date', 'SentimentLabel'])

    output_df = pd.DataFrame(columns=['Date'] + ss_average_method_names)
    output_df['Date'] = all_dates

    # Calculate the daily sentiment score for each day
    for d in all_dates:
        sent_scores_dict = {'0': {'ConfidenceSum': 0, 'Count': 0},
                            '1': {'ConfidenceSum': 0, 'Count': 0},
                            '2': {'ConfidenceSum': 0, 'Count': 0}}

        for x in range(3):
            try:
                group = split_by_date_and_label.get_group((d, x))
                count = len(group)
                sum = group['SentimentConfidence'].sum()

            except KeyError:
                count = 0
                sum = 0

            sent_scores_dict[str(x)]['Count'] = count
            sent_scores_dict[str(x)]['ConfidenceSum'] = sum

        for i, ssam in enumerate(ss_average_methods):
            ss_scores[i].append(ssam(sent_scores_dict))

        num_tweets.append(sent_scores_dict['0']['Count'] + sent_scores_dict['1']['Count'] + sent_scores_dict['2']['Count'])

    for i, ss_name in enumerate(ss_average_method_names):
        output_df[ss_name] = ss_scores[i]

    output_df[num_tweets_key] = num_tweets

    return output_df


def export_json_graph(company_name, sentiment_price_df, column_ss_keys_to_graph, to_file=False):
    spdf = sentiment_price_df

    six_colors = [
        "rgb(11, 219, 136)",
        "rgb(11, 153, 219)",
        "rgb(255, 23, 185)",

        "rgb(33, 250, 200)",
        "rgb(33, 200, 250)",
        "rgb(255, 70, 200)"
    ]

    with open('chartjsBaseGraph.Json') as f:
        json_data = json.load(f)

    datasets = []

    # default_dataset_dict = {"label": "Default",
    #                         "data": [],
    #                         "borderColor": "rgb(0, 0, 0)",
    #                         "backgroundColor": "rgba(0, 0, 0, 0.5)",
    #                         "yAxisID": "y1"}

    for i, k in enumerate(column_ss_keys_to_graph):
        # Missing values must be None for chart.js
        # -1 is currently used for No Sentiment Score so convert those to None
        data = [x if x != -1 else None for x in spdf[k].tolist()]

        # backgroundColor is borderColor with 0.5 transparency
        # If borderColor = rgb(100, 100, 100) --> backgroundColor = rgba(100, 100, 100, 0.5)
        d = {"label": k, "data": data, "borderColor": six_colors[i],
             "backgroundColor": six_colors[i][:3] + "a" + six_colors[i][3:-1] + ", 0.5)", "yAxisID": k}

        # Hide Spam-Free by default
        if 'SpamFree' in k:
            d["hidden"] = True

        json_data["options"]["scales"][k] = {"type": "linear", "position": "left", "display": False, "min": 0,
                                             "max": 100, "grid": {"drawOnChartArea": False, "tickMarkLength": 0},
                                             "ticks": {"display": False}}

        datasets.append(d)

    price_data = {"label": "Price", "data": spdf["Close"].tolist(), "borderColor": 'rgb(3, 86, 252)',
                  "backgroundColor": 'rgba(3, 86, 252, 0.5)', "yAxisID": "price", "borderWidth": 8}
    volume_data = {"label": "Volume", "type": "bar", "data": spdf["Volume"].tolist(),
                   "borderColor": 'rgb(186, 37, 219)',
                   "backgroundColor": 'rgba(186, 37, 219, 0.5)', "yAxisID": "volume", "grouped": False}
    num_tweets_data = {"label": "Tweets", "type": "bar", "data": spdf["#Tweets"].tolist(),
                       "borderColor": 'rgb(255, 170, 0)', "backgroundColor": 'rgba(225, 170, 0, 0.5)',
                       "yAxisID": 'tweets'}
    num_tweets_data_sf = {"label": "Tweets-SpamFree", "type": "bar", "data": spdf["#TweetsSpamFree"].tolist(),
                          "borderColor": 'rgb(119, 255, 0)', "backgroundColor": 'rgba(119, 255, 0, 0.5)',
                          "yAxisID": 'tweets', "hidden": True}

    datasets.append(price_data)
    datasets.append(volume_data)
    datasets.append(num_tweets_data)
    datasets.append(num_tweets_data_sf)

    json_data["data"]["datasets"] = datasets
    json_data["data"]["labels"] = spdf['Date'].tolist()

    json_data["options"]["plugins"]["title"]["text"] = company_name

    json_data["options"]["scales"]["x"]["min"] = str(spdf['Date'][0])
    json_data["options"]["scales"]["x"]["max"] = str(spdf['Date'][len(spdf) - 1])

    json_data["options"]["scales"]["price"]["min"] = (int(min(spdf['Close']) * 0.75))
    json_data["options"]["scales"]["price"]["max"] = (int(max(spdf['Close'])) * 1.25)

    json_data["options"]["scales"]["volume"]["min"] = 0
    json_data["options"]["scales"]["volume"]["max"] = (int(max(spdf['Volume'])) * 5)

    json_data["options"]["scales"]["tweets"]["min"] = 0
    json_data["options"]["scales"]["tweets"]["max"] = (int(max(spdf['#Tweets'])) * 10)

    print(json_data)

    if to_file:
        with(open(f'{company_name}Graph.json', 'w+')) as f:
            json.dump(json_data, f, indent=4)

    return json_data

def to_html(json_data, to_file=False):
    html = ""


xx = gen_all_date_based_dataframes()
k = ['Og', 'Sum', 'OgSub', 'OgSpamFree', 'SumSpamFree',
     'OgSubSpamFree']
js = [export_json_graph(n, x, k) for n, x in zip(names, xx)]

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
