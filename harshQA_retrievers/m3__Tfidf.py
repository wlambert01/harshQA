#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import os
import re
import sys
import uuid
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.feature_extraction.text as skf
from sklearn.base import BaseEstimator
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import tokenize as tkn
from nltk import pos_tag, word_tokenize
import scipy.sparse as sp
from utils.utils import convert 



# In[ ]:



class m3_Tfidf(BaseEstimator):
    """
    A scikit-learn wrapper for TfidfRetriever. Trains a tf-idf matrix from a corpus
    of documents then finds the most top_n similar documents of a given input document by
    taking the dot product of the vectorized input document and the trained tf-idf matrix.
    
    Parameters
    ----------
    ngram_range : bool, optional
        [shape of ngram used to build vocab] (the default is (1,2) (bigram))
    max_df : bool, optional
        [while building vocab delete words that have a frequency>max_df] (the default is False)
    stop_words : str, optional
        ['english is the only value accepted'] (the default is False)
    top_n : int
        maximum number of top documents to retrieve
    verbose : bool, optional
        If true, all of the warnings related to data processing will be printed.
    Attributes
    ----------
    vectorizer : TfidfVectorizer
        See https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    tfidf_matrix : sparse matrix, [n_samples, n_features]
        Tf-idf-weighted document-term matrix.
        
    Examples
    --------
    >>> retriever = m3_Tfidf(ngram_range=(1, 2), max_df=0.85, stop_words='english',lemmatize=True)
    >>> retriever.fit(X=df['content'])
    >>> doc_index=int(input('Which document do you want to query?'))
    >>> retriever.transform(X=df.loc[doc_index,'content'])
    >>> Q=str(input('Enter your question'))
    >>> closest_docs,scores = self.retriever.predict(Q)
    """

    def __init__(self,
                ngram_range=(1, 1),
                max_df=0.85,
                stop_words='english',
                verbose=False, 
                top_n=5,
                lemmatize=False,
                transform_text=True,
                save_idfs_path=None,
                save_features_path=None):

        self.ngram_range = ngram_range
        self.max_df = max_df
        self.stop_words = stop_words
        self.top_n = top_n
        self.verbose = verbose
        self.transform_text=transform_text
        self.lemmatize=lemmatize and self.transform_text
        self.stem=not lemmatize and self.transform_text
        if self.stem: self.stemmer=PorterStemmer()
        else: self.lemmatizer=WordNetLemmatizer() 
        self.stop_words_list=[self.tokenize(word)[0] for word in list(skf._check_stop_list('english'))]
        self.idfs_path=save_idfs_path
        self.features_path=save_features_path

    def stem_tokens(self,tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed
    
    def lemmatize_tokens(self,tokens,lemmatizer):
        lemmas=[]
        for item in tokens:
            for word, tag in pos_tag(word_tokenize(item)):
                wntag = tag[0].lower()
                wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
                if not wntag:
                    lemma = word
                else:
                    lemma = lemmatizer.lemmatize(word, wntag)
            
            lemmas.append(lemma)
        return lemmas

    def tokenize(self,text):
        tokens = nltk.word_tokenize(text)
        if self.lemmatize:
        #stems = self.stem_tokens(tokens, self.stemmer)
            lemmas=self.lemmatize_tokens(tokens,self.lemmatizer)
            return lemmas
        elif self.stem:
            stems = self.stem_tokens(tokens, self.stemmer)
            return stems
        else:
            return tokens
        
    def fit(self, X, y=None): #generate features and return tfidf scores matrix 

        if self.idfs_path!=None and self.features_path!=None:

            if os.path.exists(self.idfs_path) and os.path.exists(self.features_path):
                self.vectorizer=TfIdf_fromcheckpoint(ngrams=self.ngram_range,
                                              maxdf=self.max_df,
                                       stopwords=self.stop_words_list,
                                       tokenizer_=self.tokenize,
                                       path_idfs=self.idfs_path,
                                       path_json_voc=self.features_path)

            else: 
                self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range,
                                          max_df=self.max_df,
                                   stop_words=self.stop_words_list,
                                   tokenizer=self.tokenize)
                self.vectorizer.fit(X)
            
                np.save(self.idfs_path,self.vectorizer.idf_)
                with open(self.features_path, 'w') as outfile:
                    json.dump(self.vectorizer.vocabulary_,outfile,default=convert)
        else:
            
            self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range,
                                          max_df=self.max_df,
                                   stop_words=self.stop_words_list,
                                   tokenizer=self.tokenize)
            self.vectorizer.fit(X)


        return self
    
    def transform(self,X,y=None):

        self.tfidf_matrix=self.vectorizer.transform(X)
        return self
    
    def predict(self, X):
        tfidf_matrix=self.tfidf_matrix
        #cherche les querries les plus proches de chaque sentence
        t0 = time.time()
        question_vector = self.vectorizer.transform([X])
        data=tfidf_matrix.dot(question_vector.T).toarray()
        scores = pd.DataFrame(data,index=range(len(data)))
        closest_docs_indices = scores.sort_values(by=0, ascending=False).index[:self.top_n].values

        # inspired from https://github.com/facebookresearch/DrQA/blob/50d0e49bb77fe0c6e881efb4b6fe2e61d3f92509/scripts/reader/interactive.py#L63
        if self.verbose:
            print('Time: {} seconds'.format(round(time.time() - t0, 5)))

        return closest_docs_indices,scores

class TfIdf_fromcheckpoint(TfidfVectorizer):
    def __init__(self,ngrams, maxdf, stopwords, tokenizer_, path_idfs=None,path_json_voc=None):
        super().__init__(ngram_range=ngrams, max_df=maxdf, stop_words=stopwords, tokenizer=tokenizer_)
        assert os.path.exists(path_idfs), "problem with tfidf model path (.npy file)"
        assert os.path.exists(path_json_voc), "problem with json voc (.json file)"
   
        if os.path.exists(path_idfs):
            self.idf_= np.load(path_idfs)
        if os.path.exists(path_json_voc):
            vocabulary = json.load(open(path_json_voc, mode = 'rb'))
            self.vocabulary_ = vocabulary
        self._tfidf._idf_diag = sp.spdiags(self.idf_,diags = 0, m = len(self.idf_), n = len(self.idf_))







