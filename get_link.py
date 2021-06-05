import pickle
import numpy as np 
import pandas as pd 
from gensim.parsing.preprocessing import remove_stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
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


def clean_data(texts):
    '''處理stop words'''
    cleaned_corpus = []
    for c in texts:
        cleaned_corpus.append(remove_stopwords(c))
    return cleaned_corpus


def get_TFIDF(texts, vectorizer):
    '''TFIDF training'''
    # vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts)
    return tfidf.toarray()


def get_scores(tfidf):
    scores = np.zeros((len(tfidf), len(tfidf)))
    for i in range(len(tfidf)):
        for j in range(len(tfidf)):
            scores[i,j] = cos_sim(tfidf[i], tfidf[j])
    scores = np.where(scores > 0.999, 0, scores)
    return scores



# def get_scores(tfidf):
#     '''得到每個文本相對應的相似度'''
#     tfidfs = []
#     for i in range(len(corpus)):
#         tfidfs.append(cosine(tfidf[i:i+1], tfidf))
#     tfidfs = np.stack(tfidfs).squeeze()
#     tfidfs = np.where(tfidfs>0.99, 0, tfidfs).flatten()
#     return tfidfs


def get_links(corpus_count, scores, max_count=3):
    '''得到前max_count的連結, 1 ---> 2 表示文章是從第1篇連到第2篇'''
    indexs = np.argsort(scores.flatten())[-1:-max_count-1:-1]
    
    for i in indexs:
        s, d = i//corpus_count +1, i%corpus_count
        print(f'{s} ----> {d}')



if __name__ == '__main__':
    csv_path = 'data/result.json'
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    corpus = get_corps(csv_path)
    cleaned_corpus = clean_data(corpus)
    tfidf = get_TFIDF(cleaned_corpus, vectorizer)
    scores = get_scores(tfidf)
    get_links(len(corpus), scores)
