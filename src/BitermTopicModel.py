import bitermplus as btm
import numpy as np
import pickle
import os
import pandas as pd
from utilities import Utils
from NLPSentimentCalculations import NLPSentimentCalculations as NLPSC


# from nltk import Tree
# import tmplot as tmp


def process_tweets(name, same_file=True, bigram_cutoff=1.0):
    df = pd.read_csv(name + 'FinalData.csv')
    if 'full_text' not in df.keys():
        df = Utils.parse_json_tweet_data(df, ['full_text'])

    print(name + 'FinalData.csv')
    print("Sanitizing")
    if 'san_full_text' not in df.keys():
        ft = df['full_text'].tolist()
        san = [NLPSC.sanitize_text_string(t) for t in ft]
        df['san_full_text'] = san

    pre_san_size = len(df)
    print(f"Pre-San Size: {pre_san_size}")

    df = df.loc[df['san_full_text'] != '']
    df = df[df['san_full_text'].apply(lambda x: isinstance(x, str))]

    print(f"Post-San Size: {len(df)}")
    print(f"Lost: {pre_san_size - len(df)}")
    print()

    print("Grouping Bigrams")
    bigrams_y = NLPSC.group_bigrams(df['san_full_text'].tolist(), cutoff=bigram_cutoff)
    bi_df = pd.DataFrame(bigrams_y[1].items(), columns=['Original', 'XBigram'])
    df['B-1'] = bigrams_y[0]

    print("Saving")
    if same_file:
        new_path = name + 'FinalData.csv'
        df.to_csv(new_path, index=False)

        bigram_path = name + 'BigramKeys.csv'
        bi_df.to_csv(bigram_path, index=False)

    return df


def prep_btm(texts):
    X, vocabulary, vocab_dict = btm.get_words_freqs(texts)
    docs_vec = btm.get_vectorized_docs(texts, vocabulary)
    biterms = btm.get_biterms(docs_vec)

    return X, vocabulary, vocab_dict, docs_vec, biterms


def train_btm(PrepData, topics_n, iterations, background):
    X, vocabulary, vocab_dict, docs_vec, biterms = PrepData

    model = btm.BTM(
        X, vocabulary, seed=12345, T=topics_n, M=20, alpha=50 / topics_n, beta=0.01, has_background=background)
    model.fit(biterms, iterations=iterations)
    p_zd = model.transform(docs_vec)
    return model


def prep_train_save_btm(name, iterations, top_keywords=14, tweets_per_topic=3500):
    # Load Bigramed csv and preprocess
    df = pd.read_csv(name + 'FinalData.csv')

    texts = df['B-1'].tolist()
    prep_data = prep_btm(texts)

    # Set Background and Number of Topics based on data
    if len(texts) < 10000:
        background = False
    else:
        background = True

    # Approximately 1 topic per 3500 Tweets
    n = max(4, len(texts) // tweets_per_topic)

    # Train model and get top keywords (using top-sensitive mix)
    model = train_btm(prep_data, n, iterations, background)
    d = get_top_sensitive_keyword_mix(model, top_keywords)

    # Load Bigram_keys to restore words (jackxdorsey --> jack dorsey)
    bigram_keys = pd.read_csv(name + 'BigramKeys.csv')
    bigram_map = {k: v for k, v in zip(bigram_keys['XBigram'].tolist(), bigram_keys['Original'].tolist())}

    # Turn Bigrams back into words
    d1 = {}
    for k, v in d.items():
        new_terms = []
        for term in v:
            if term in bigram_map.keys():
                new_terms.append(bigram_map[term])
            else:
                new_terms.append(term)
        d1[k] = new_terms

    # Make topic df
    topic_df = pd.DataFrame(d1)
    topic_df = topic_df.rename({x: f'Topic #{x}' for x in topic_df.columns}, axis=1)
    topic_df['Nth Top Keyword'] = list(range(1, top_keywords + 1))
    topic_df = topic_df[['Nth Top Keyword'] + [f'Topic #{x}' for x in range(0, n)]]

    df['Topic #'] = model.labels_

    # if name != 'Trump':
    #     df = df[
    #         ['Tweet id', 'User id', 'Screen name', 'Topic #', 'SpamScore', 'SpamScoreRaw',
    #          'SentimentScore', 'SentimentScoreRaw', 'clean_full_text', 'full_text', 'san_full_text', 'B-1',
    #          'Search term', 'json']]
    #
    # else:
    #     df = df[
    #         ['Tweet id', 'Topic #', 'SpamScore', 'SpamScoreRaw',
    #          'SentimentScore', 'SentimentScoreRaw', 'clean_full_text', 'full_text', 'san_full_text',
    #          'B-1']]

    df = df.sort_values('Topic #', axis=0)

    # Write Tweet data to csv and Topic df to csv
    df.to_csv(f'{name}FinalData.csv', index=False)
    topic_df.to_csv(f'{name}Topics.csv', index=False)

    # Save model to pickle
    with open(f"{name}Model.pickle", "wb") as file:
        pickle.dump(model, file)

    return model, topic_df, df


def run_all():
    dfa = pd.read_csv('../data/TestingData/AppleBigrams.csv')
    dft = pd.read_csv('../data/TestingData/TrumpBigrams.csv')

    vers = ['b-1.5']
    prepa = [prep_btm(dfa[v].tolist()) for v in vers]
    prept = [prep_btm(dft[v].tolist()) for v in vers]

    modelsa = [train_btm(P, 20, 400, True) for P in prepa]
    modelst = [train_btm(P, 20, 400, True) for P in prept]

    return (modelsa, modelst)


def printd(d):
    [print(k, ':', v) for k, v in d.items()]


def run_btm(texts, topics_n):
    # PREPROCESSING
    # Obtaining terms frequency in a sparse matrix and corpus vocabulary
    X, vocabulary, vocab_dict = btm.get_words_freqs(texts)

    # tf = np.array(X.sum(axis=0)).ravel()

    # Vectorizing documents
    docs_vec = btm.get_vectorized_docs(texts, vocabulary)

    # docs_lens = list(map(len, docs_vec))

    # Generating biterms
    biterms = btm.get_biterms(docs_vec)

    # INITIALIZING AND RUNNING MODEL
    # model = btm.BTM(
    #     X, vocabulary, 8, unsigned_int_seed=12321, int_M=20, double_alpha=50 / 8, double_beta=0.01)
    model = btm.BTM(
        X, vocabulary, seed=12345, T=topics_n, M=20, alpha=50 / topics_n, beta=0.01)
    model.fit(biterms, iterations=20)
    p_zd = model.transform(docs_vec)

    print(model.perplexity_)
    print(model.coherence_)
    print(model.labels_)

    return model


#
# def tmplot(model, texts):
#     tmp.report(model=model, docs=texts)


def get_top_keywords(model, n):
    if type(model) == pd.DataFrame:
        x = model
    else:
        x = model.df_words_topics_

    res_keys = [i for i in range(x.shape[1])]
    res_data = []
    for i in range(x.shape[1]):
        res_data.append(list(x.nlargest(n, i).index))

    return dict(zip(res_keys, res_data))


def get_sensitive_keywords(model, n):
    if type(model) == pd.DataFrame:
        x = model
    else:
        x = model.df_words_topics_

    y = x.div(x.sum(axis=1), axis=0)
    return get_top_keywords(y, n)


def get_top_sensitive_keyword_mix(model, n):
    if type(model) == pd.DataFrame:
        x = model
    else:
        x = model.df_words_topics_

    y = x.div(x.sum(axis=1), axis=0)
    z = x.mul(y)
    return get_top_keywords(z, n)


def process(csv, num_topics, num_words):
    df = process_tweets(csv)
    model = run_btm(df['san_full_text'].tolist(), num_topics)
    return get_top_keywords(model, num_words)

# df = pd.read_csv('../data/TestingData/BTMDemo.csv')
# san = df['san_full_text'].tolist()
# model = run_btm(san, 4)
# d = get_top_keywords(df, model, 6)

# # IMPORTING DATA
# df = pd.read_csv(
#     'dataset/SearchSnippets.txt.gz', header=None, names=['texts'])
# texts = df['texts'].str.strip().tolist()


# # METRICS
# perplexity = btm.perplexity(model.matrix_topics_words_, p_zd, X, 8)
# coherence = btm.coherence(model.matrix_topics_words_, X, M=20)
# or


# # LABELS
# print(model.labels_)
# # or
# btm.get_docs_top_topic(texts, model.matrix_docs_topics_)
