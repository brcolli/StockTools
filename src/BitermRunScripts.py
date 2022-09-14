import pandas as pd
import os
from utilities import Utils

names = ['Apple', 'CallOptions', 'Ethereum', 'FormulaOne', 'InterestRates', 'JackDorsey', 'NancyPelosi', 'SupplyChain', 'Tesla', 'Trump', 'TruthSocial', 'WWDC']
# names = ['Ethereum']

os.chdir('../data/TestingData')


def runk():
    # merge_sentiment_data()
    os.chdir('3.5kTopics')
    average_sentiment_spam_data()
    os.chdir('../2kTopics')
    average_sentiment_spam_data()


def drop_spam():
    for n in names:
        print(n)
        df = pd.read_csv(n+'FinalData.csv')
        print(f'Len With Spam: {len(df)}')

        df = df.dropna()
        df = df[df['SpamScore'] == 0]

        print(f'Len With no Spam: {len(df)}')
        print(f"All Bigrams are text: {all(isinstance(x, str) for x in df['B-1'])}")
        print(f"All Bigrams are non-empty: {all(len(x) > 0 for x in df['B-1'])}")

        df.to_csv(n+'FinalData.csv', index=False)


def get_whole_dataset_averages():
    df = pd.DataFrame(columns=['Metric'] + names)
    df['Metric'] = ['# Tweets', '% Clean', '% Spam', 'Spam Confidence', '% Positive', '% Negative', 'Sentiment Confidence']

    sgp = safe_get_percent
    gsc = get_single_count

    for n in names:
        data = pd.read_csv(n+'FinalData.csv')
        nt = len(data)
        spam = data.value_counts('SpamScore')
        sent = data.value_counts('SentimentScore')

        pclean = sgp(gsc(spam, 0), nt)
        pspam = sgp(gsc(spam, 1), nt)
        confspam = data['SpamScoreRaw'].mean()

        ppos = sgp(gsc(sent, 0), nt)
        pneg = sgp(gsc(sent, 1), nt)
        confsent = data['SentimentScoreRaw'].mean()

        column = [int(nt), pclean, pspam, confspam, ppos, pneg, confsent]
        df[n] = column

    return df


def merge_sentiment_data():
    for n in names:
        sent = pd.read_csv(f'MVPBenLabels/{n}DataLabeledCleaned.csv')

        for v in ['3.5kTopics', '2kTopics']:
            data = pd.read_csv(f'{v}/{n}Data.csv')

            data['SpamScore'] = sent['SpamScore']
            data['SpamScoreRaw'] = sent['SpamScoreRaw']
            data['SentimentScore'] = sent['SentimentScore']
            data['SentimentScoreRaw'] = sent['SentimentScoreRaw']
            data['clean_full_text'] = sent['full_text']

            if n != 'Trump':
                data = data[['Tweet id', 'User id', 'Screen name', 'Topic #', 'SpamScore', 'SpamScoreRaw',
                             'SentimentScore', 'SentimentScoreRaw', 'clean_full_text', 'full_text', 'Search term',
                             'json']]
            else:
                data = data[['Tweet id', 'Topic #', 'SpamScore', 'SpamScoreRaw',
                             'SentimentScore', 'SentimentScoreRaw', 'clean_full_text', 'full_text']]

            data.to_csv(f'{v}/{n}FinalData.csv', index=False)


def get_count(counts, k1, k2):
    try:
        c = counts[k1][k2]

    except KeyError:
        c = 0

    return c


def get_single_count(counts, k1):
    try:
        c = counts[k1]
    except KeyError:
        c = 0

    return c


def safe_get_percent(n, total):
    if total == 0:
        return 0
    else:
        return n / total


def average_sentiment_spam_data(average_type=None):
    if average_type is None:
        average_type = ['SpamScore', 'SentimentScore']

    for n in names:
        data = pd.read_csv(n+'FinalData.csv')
        topics = pd.read_csv(n+'Topics.csv')

        agg_keys = topics.keys()[1:]
        topic_nums = [x for x in range(len(agg_keys))]

        counts = data['Topic #'].value_counts()
        counts_list = ['# Tweets'] + [get_single_count(counts, k) for k in topic_nums]
        topics.loc[len(topics.index)] = counts_list

        sgp = safe_get_percent

        spam_counts = data.value_counts(['Topic #', 'SpamScore'])
        spam_list = ['% Spam'] + [sgp(get_count(spam_counts, x, 1), get_single_count(counts, x)) for x in topic_nums]
        topics.loc[len(topics.index)] = spam_list

        sentiment_counts = data.value_counts(['Topic #', 'SentimentScore'])
        sentiment_list = ['% Positive Sentiment'] +\
                         [sgp(get_count(sentiment_counts, x, 0), get_single_count(counts, x)) for x in topic_nums]
        topics.loc[len(topics.index)] = sentiment_list

        topics.to_csv(n+'Topics.csv', index=False)
        # return topics


def replace_xs():
    for n in names:
        print(n)
        df = pd.read_csv(n+'Topics.csv')
        bg = pd.read_csv(n+'BigramKeys.csv')

        bigram_keys = pd.read_csv(n + 'BigramKeys.csv')
        bigram_map = {k: v for k, v in zip(bigram_keys['XBigram'].tolist(), bigram_keys['Original'].tolist())}

        # Turn Bigrams back into words
        for k in df.keys():
            df[k] = [bigram_map[x] if x in bigram_map.keys() else x for x in df[k].tolist()]

        df.to_csv(n+'Topics.csv', index=False)

#
# os.chdir('../data/TestingData/Per2kTopics')
# replace_xs()


def fix():
    for n in names:
        print(n)

        original = pd.read_csv(n+'.csv')
        print(f'Original: {len(original)}')

        data = pd.read_csv(n+'Data.csv')
        print(f'Data: {len(data)}')

        bigramed = pd.read_csv(n+'Bigramed.csv')
        print(f'Bigramed: {len(bigramed)}')

        print("Integerizing Topic #")
        data['Topic #'] = [int(d) for d in data['Topic #'].tolist()]

        print("Restoring Ids")
        bft = bigramed['full_text'].tolist()
        indices = [bft.index(f) for f in data['full_text']]
        real_ids = bigramed['Tweet id'].tolist()
        restore_ids = [real_ids[i] for i in indices]
        restore_ids = [int(r) for r in restore_ids]

        data['Tweet id'] = restore_ids

        print("Restoring Indices")
        original_ids = original['Tweet id'].tolist()
        original_indices = [original_ids.index(x) for x in restore_ids]

        print("Setting column values")
        if n != 'Trump':
            keys = ['User id', 'Screen name', 'Search term', 'json']
            for k in keys:
                original_list = original[k].tolist()
                data[k] = [original_list[i] for i in original_indices]

            data = data[['Tweet id', 'User id', 'Screen name', 'Topic #', 'Sentiment score', 'full_text', 'Search term', 'json']]

        else:
            data = data[['Tweet id', 'Topic #', 'Sentiment score', 'full_text']]

        print()
        print('Validating')
        if n != 'Trump':
            json_list = data['json'].tolist()
            json_ids = [int(j[55:j.find(',', 55)]) for j in json_list]

            json_full_text = Utils.parse_json_tweet_data(data[['json']], ['full_text'])['full_text'].tolist()

            id_match = all(x == y for x, y in zip(data['Tweet id'].tolist(), json_ids))
            print(f"Ids match: {id_match}")

            ft_match = all(x == y for x, y in zip(data['full_text'].tolist(), json_full_text))
            print(f"Full text match: {ft_match}")

            if id_match and ft_match:
                data.to_csv(n + 'Data.csv', index=False)

        else:
            data.to_csv(n + 'Data.csv', index=False)

        print('----------------------------------------------')
        print()


# os.chdir('../data/TestingData')
# fix()
# os.chdir('Per2kTopics')
# fix()



# import pandas as pd
#
# df = pd.read_csv('../data/SurveyResponses/Labels.csv')
#
# df.loc[(df['Type2'] == 'X'), 'Type2']=-1
# df['Type2'] = df['Type2'].fillna(-1)
# df['Type'] = df['Type'].fillna(-1)
# df = df[df.Type > 2]
#
# dup = df[df['Type2'] != -1]
# dup = dup.drop('Type', axis=1)
# dup = dup.rename(columns={'Type2': 'Type'})
#
# df = df.drop('Type2', axis=1)
#
# df2 = pd.concat([df, dup])
# df2['Type'] = [int(x) for x in df2['Type'].tolist()]
# df2 = df2.sort_values(['Type'], ascending=False)
# df2 = df2[df2['Type'] != 7]
#
# labels = {3: 'misc', 4: 'misc', 5: 'indecisive', 6: 'decisive', 7: None, 8: 'indecisive', 9: 'rejection', 10: 'misc'}
# df2['label'] = df2['Type'].map(labels)
# df2 = df2.sort_values(['label'])
# labels_raw = df2['label'].unique().tolist()
# split_dfs = []
# for l in labels_raw:
#     split_dfs.append(df2.loc[df2.label == l])
#
# demographic_keys = ['Household Income', 'Age', 'Gender']
#
# data = []
# for d, l in zip(split_dfs, labels_raw):
#     print(f'{l}\n')
#     for k in demographic_keys:
#         print(f'{k};Count')
#         data = d[k].value_counts().sort_index()
#         for key in data.keys():
#             print(f'{key};{data[key]}')
#         print()
#     print()
#
#
#
# # labeler.tweet_csv[label_column].value_counts()
#
# # df2.loc[df2['Label'] in [3, 4, 10], 'Type'] = 'misc'
# # df2.loc[df2['Type'] == 9] = 'rejection'
#
#
# # df2.set_index(keys=['Type'], drop=False, inplace=True)
