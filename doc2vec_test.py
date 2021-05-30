import numpy as np 
import pandas as pd 
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


if __name__ == '__main__':
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
    print(documents)