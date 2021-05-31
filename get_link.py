import numpy as np 
import pandas as pd 
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_corps(json_path):
    '''用description來做'''
    with open(json_path, 'rb') as f:
        df = pd.read_json(f)
    corpus = []
    for i, row in df.iterrows():
        corpus.append(row['description'])
    return corpus


def clean_data(texts):
    '''處理stop words'''
    cleaned_corpus = []
    for c in texts:
        cleaned_corpus.append(remove_stopwords(c))
    return cleaned_corpus


def get_TFIDF(texts):
    '''TFIDF training'''
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts)
    return tfidf


def get_scores(tfidf):
    '''得到每個文本相對應的相似度'''
    tfidfs = []
    for i in range(len(corpus)):
        tfidfs.append(cosine_similarity(tfidf[i:i+1], tfidf))
    tfidfs = np.stack(tfidfs).squeeze()
    tfidfs = np.where(tfidfs>0.99, 0, tfidfs).flatten()
    return tfidfs


def get_links(corpus_count, tfidfs, max_count=3):
    '''得到前max_count的連結, 1 ---> 2 表示文章是從第1篇連到第2篇'''
    indexs = np.argsort(tfidfs)[-1:-max_count-1:-1]
    
    for i in indexs:
        s, d = i//corpus_count +1, i%corpus_count
        print(f'{s} ----> {d}, {i}')



if __name__ == '__main__':
    csv_path = 'data2.csv'
    corpus = get_corps(csv_path)
    cleaned_corpus = clean_data(corpus)
    tfidf = get_TFIDF(cleaned_corpus)
    tfidfs = get_scores(tfidf)
    get_links(len(corpus), tfidfs)
