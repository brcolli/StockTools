import yake
import wikipediaapi
import wordcloud
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

wiki = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)

language = "en"
max_ngram_size = 3
deduplication_threshold = 0.5
numOfKeywords = 20

custom_kw = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords)


def wiki_to_file(title):
    page = wiki.page(title)
    txt = page.text
    with open(f'data/Keywords/{title}-Wikipedia.txt', 'x', encoding='utf-8') as f:
        f.write(txt)
    return txt


def analyze_txt(text):
    return custom_kw.extract_keywords(text)


def analysis_to_dataframe(data, title):
    df = pd.DataFrame(data, columns=['N-Gram', 'Weight'])
    y = [1 / max(0.000000001, x) for x in df['Weight']]
    df['Frequency Score'] = y
    df.to_csv(f'data/Keywords/{title}-Data.csv', index=False)
    return df


def make_word_cloud(df, name, w=1200, h=600):
    wc = wordcloud.WordCloud(background_color="white", contour_width=3, contour_color='steelblue', width=w,
                             height=h)

    keys = [x.replace(' ', '-') for x in df['N-Gram']]
    vals = list(df['Frequency Score'])
    freq_dict = dict(zip(keys, vals))

    wc.generate_from_frequencies(freq_dict)
    wc.to_file(f'data/Keywords/{name}-WordCloud.png')

    image = Image.open(f'data/Keywords/{name}-WordCloud.png')
    image.show()
    return image


def make_freq_graph(df, name):
    x = df['N-Gram']
    x_pos = [i for i, _ in enumerate(x)]
    y = list(df['Frequency Score'])
    total = sum(y)
    y = [y1/total * 100 for y1 in y]

    plt.bar(x, y)
    plt.xlabel('N-Grams')
    plt.ylabel('Frequency Score')
    plt.title(name)
    plt.xticks(x_pos, x)

    plt.show()
    plt.savefig(f'data/Keywords/{name}-Graph.png')
    return plt


def pie_chart(df, name, trim=10):
    if trim > len(df):
        trim = len(df)

    labels = df['N-Gram'][0:trim]
    y = list(df['Frequency Score'])[0:trim]
    total = sum(y)
    y = [y1 / total * 100 for y1 in y]
    explode = [max(y1/100 - 0.5, 0) for y1 in y]
    plt.pie(y, labels=labels, explode=explode)

    print(labels, y, explode)
    plt.title(name)
    plt.show()
    plt.savefig(f'data/Keywords/{name}-PieChart.png')
    return plt

# def to_pdf(image, data)


def pull_and_analyze(title):
    txt = wiki_to_file(title)
    data = analyze_txt(txt)
    df = analysis_to_dataframe(data, title)
    wc = make_word_cloud(df, title)
    pie = pie_chart(df, title)
