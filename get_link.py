import pickle
import numpy as np 
import pandas as pd 
from numpy import dot
from numpy.linalg import norm


def cos_sim(a,b):
    return dot(a, b)/(norm(a)*norm(b))


def get_corps(json_path):
    '''用description來做'''
    with open(json_path, 'rb') as f:
        df = pd.read_json(f)
    corpus = []
    for i, row in df.iterrows():
        corpus.append(row['description'])
    return corpus


def get_TFIDF(texts, vectorizer):
    '''TFIDF training'''
    tfidf = vectorizer.fit_transform(texts)
    return tfidf.toarray()


def get_scores(tfidf):
    scores = np.zeros((len(tfidf), len(tfidf)))
    for i in range(len(tfidf)):
        for j in range(len(tfidf)):
            scores[i,j] = cos_sim(tfidf[i], tfidf[j])
    scores = np.where(scores > 0.999, 0, scores)
    return scores


def get_links(corpus_count, scores, max_count=3):
    scores = scores.flatten()
    index_corpus = []
    index_link = []
    links = []
    for i in range(corpus_count):
        for j in range(corpus_count):
            index_corpus.append(i)
            index_link.append(j)

    df = pd.DataFrame(columns=['scores', 'index_corpus', 'index_link'])
    df['scores'] = scores
    df['index_corpus'] = index_corpus
    df['index_link'] = index_link
    df = df.sort_values(by='scores', ascending=False).values
    for i in range(max_count):
        links.append([df[2*i, 1].astype(int), df[2*i, 2].astype(int)])
    return links


if __name__ == '__main__':
    csv_path = 'data/result.json'
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    corpus = get_corps(csv_path)
    tfidf = get_TFIDF(corpus, vectorizer)
    scores = get_scores(tfidf)
    links = get_links(len(corpus), scores)
