import os
import yake
import wikipediaapi
import wordcloud
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from keybert import KeyBERT
from textwrap import wrap


class Extractor:
    def __init__(self, number_keywords=15, min_ngram=1, max_ngram=2, diversity_level=0.5,
                 match_language_structure=True, file_path='../data/Keywords/'):
        self.number_keywords = number_keywords
        self.ngram_range = (min_ngram, max_ngram)
        self.diversity_level = diversity_level
        self.match_lang = match_language_structure
        self.path = file_path

        self.deduplication_threshold = 1 - diversity_level

        self.wiki = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)

        self.yake_kw = yake.KeywordExtractor(lan="en", n=self.ngram_range[1], dedupLim=self.deduplication_threshold,
                                             top=self.number_keywords)
        self.bert_kw = KeyBERT()

    def relaunch_yake(self):
        self.yake_kw = yake.KeywordExtractor(lan="en", n=self.ngram_range[1], dedupLim=self.deduplication_threshold,
                                             top=self.number_keywords)

    def bert_extract(self, text):
        return self.bert_kw.extract_keywords(text, keyphrase_ngram_range=self.ngram_range, stop_words='english',
                                             use_mmr=True, diversity=self.diversity_level, top_n=self.number_keywords)

    def bert_to_df(self, data):
        df = self.analysis_to_dataframe(data)
        raw = df['Raw Score'].tolist()
        total = sum(raw)
        weighted = [r * 100 for r in raw]
        percent_score = [w / total for w in weighted]
        df['Normalized Score'] = weighted
        df['Percent Score'] = percent_score
        df.sort_values('Normalized Score', ascending=False, inplace=True)
        return df

    def yake_extract(self, text):
        return self.yake_kw.extract_keywords(text)

    def yake_to_df(self, data):
        df = self.analysis_to_dataframe(data)
        raw = df['Raw Score'].tolist()
        raw = [1 / r for r in raw]
        total = sum(raw)
        weighted = [100 * (r/total) for r in raw]
        df['Normalized Score'] = weighted
        df.sort_values('Normalized Score', ascending=False, inplace=True)
        return df

    @staticmethod
    def analysis_to_dataframe(data):
        df = pd.DataFrame(data, columns=['N-Gram', 'Raw Score'])
        return df

    def pull_file(self, title):
        if os.path.isfile(f'{self.path+title}.txt'):
            with open(f'{self.path+title}.txt', 'r', encoding='utf-8') as f:
                txt = f.read()
            return txt
        else:
            return -1

    def wiki_to_file(self, title):
        if os.path.isfile(f'{self.path+title}-Wikipedia.txt'):
            with open(f'{self.path+title}-Wikipedia.txt', 'r', encoding='utf-8') as f:
                txt = f.read()
            return txt

        page = self.wiki.page(title)
        txt = page.text

        if not txt:
            return -1

        with open(f'{self.path}{title}-Wikipedia.txt', 'x', encoding='utf-8') as f:
            f.write(txt)
        return txt

    def make_word_cloud(self, df, w=1200, h=600):
        wc = wordcloud.WordCloud(background_color="white", contour_width=3, contour_color='steelblue', width=w,
                                 height=h)

        keys = [x.replace(' ', '-') for x in df['N-Gram']]
        vals = list(df['Normalized Score'])
        freq_dict = dict(zip(keys, vals))
        wc.generate_from_frequencies(freq_dict)

        return wc

    def pie_chart(self, df, name, trim=10):
        if trim > len(df):
            trim = len(df)

        labels = df['N-Gram'][0:trim]
        y = list(df['Normalized Score'])[0:trim]
        total = sum(y)
        y = [y1 / total * 100 for y1 in y]
        explode = [max(y1 / 100 - 0.5, 0) for y1 in y]

        plt.clf()
        plt.pie(y, labels=labels, explode=explode, autopct='%1.0f%%')
        print(labels, y, explode)

        title = "\n".join(wrap(name, 60))
        plt.title(title)

        return plt

    def pull_and_analyze(self, title, source='local', search_term='', algorithms=None, word_cloud=False,
                         pie_chart=True):

        if algorithms is None:
            algorithms = ['yake', 'bert']

        if source == 'local':
            txt = self.pull_file(title)

        elif source == 'wiki':
            if search_term:
                txt = self.wiki_to_file(search_term)
            else:
                txt = self.wiki_to_file(title)

        else:
            print(f"Source {source} not found")
            return -1

        if txt == -1:
            print('Lookup term not found')
            return -1

        results = []
        for alg in algorithms:
            if alg == 'bert':
                data = self.bert_extract(txt)
                df = self.bert_to_df(data)
            elif alg == 'yake':
                data = self.yake_extract(txt)
                df = self.yake_to_df(data)
            else:
                print(f'No matching type of algorithm: {alg} found')
                continue

            df.to_csv(f'{self.path}{title}-{alg}-Data.csv', index=False)

            if word_cloud:
                wc = self.make_word_cloud(df)
                wc.to_file(f'{self.path}{title}-{alg}-WordCloud.png')
                image = Image.open(f'{self.path}{title}-{alg}-WordCloud.png')
                image.show()

            if pie_chart:
                pie = self.pie_chart(df, f'{title}-{alg}')
                plt.savefig(f'{self.path}{title}-{alg}-PieChart.png')
                image2 = Image.open(f'{self.path}{title}-{alg}-PieChart.png')
                image2.show()

            results.append(df)

        return results


ex = Extractor()

# def make_freq_graph(df, name):
#     x = df['N-Gram']
#     x_pos = [i for i, _ in enumerate(x)]
#     y = list(df['Frequency Score'])
#     total = sum(y)
#     y = [y1 / total * 100 for y1 in y]
#
#     plt.bar(x, y)
#     plt.xlabel('N-Grams')
#     plt.ylabel('Frequency Score')
#     plt.title(name)
#     plt.xticks(x_pos, x)
#
#     plt.show()
#     plt.savefig(f'data/Keywords/{name}-Graph.png')
#     return plt


# def to_pdf(image, data)
