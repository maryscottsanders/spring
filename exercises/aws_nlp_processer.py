
# imports
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.corpora import Dictionary
from nltk.stem import WordNetLemmatizer, SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter, OrderedDict
#import numpy as np
from sklearn.externals import joblib

def lemmatize_stemming(text):
    '''
    Function to perform the lemmatization and stemming
    '''
    stemmer = SnowballStemmer("english")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Tokenize and lemmatize
def preprocess(text):
    '''
    Function to perform preprocessing: remove stopwords, short words, lemmatization, 
    and stemming. Returns doc represented as a string after appending tokens.
    '''
    result=[]
    for token in simple_preprocess(text):
        if token not in STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return ' '.join(result)

def gen_processed_docs(data):
    '''
    Appends preprocessed docs together to make a list of strings.
    '''
    processed_docs = []
    for doc in data:
        processed_docs.append(preprocess(doc))
    return processed_docs

def gen_bow_corpus(data):
    '''
    Generates and saves a countvectorizer in addition to returning the bow_corpus.
    '''
    processed_docs = gen_processed_docs(data)
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', lowercase=False)
    tf_vectorizer = tf_vectorizer.fit(processed_docs)
    joblib.dump(tf_vectorizer, 'tf_vectorizer.pkl')
    bow_corpus = tf_vectorizer.transform(processed_docs)
    return bow_corpus