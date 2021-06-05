import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


if __name__ == '__main__':
    vectorizer = cosine_similarity()
    
    with open('cosine_similarity.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    