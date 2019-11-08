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
import nltk
import torch
from sklearn.base import BaseEstimator
from sentence_transformers import SentenceTransformer
import numpy as np



# In[ ]:


class m2_Bert(BaseEstimator):

    """
    A scikit-learn estimator for Bert Retriever. Update the vocab with the corpus content and
    create an embedding matrix by encoding sentences with Bert Large pretrained.
    Then finds the most top_N similar documents of a given input document by
    taking the dot product of the vectorized input document and the embedding matrix.
    
    Parameters
    ----------
    top_n : int
        maximum number of top articles to retrieve

    Examples
    --------
    >>> retriever = m2_Bert(top_n=5)
    
    >>> doc_index=int(input('Which document do you want to query?'))
    >>> retriever.transform(X=df.loc[doc_index,'content'])
    >>> closest_docs,scores = self.retriever.predict("Has the company formalized  a strategy to reduce its greenhouse gas  emissions?")
    
    """ 
    
    def __init__(self,top_n=5):
        self.top_n = top_n
        self.bert=SentenceTransformer('bert-large-nli-stsb-mean-tokens')
        
    def fit(self,X,y=None):
        """ X: any iterable which contains words to finetune vocabulary """
        return self
    
    def transform(self,X,y=None):
        """ X: any iterable which contains sentences to embed """
        self.embeddings = self.bert.encode(list([s for s in X ]))
        self.reduced_embeddings = np.apply_along_axis(lambda v: self.normalize(v), 1,self.embeddings)
        return self
    
    def normalize(self,array):
        return array/np.linalg.norm(array)
    
    def predict(self,X):
        
        question=self.bert.encode([X])
        encoded_question=self.normalize(question)
        self.reduced_question=self.normalize(question)
        
        reduced_embeddings=self.reduced_embeddings
        
        data=reduced_embeddings.dot(self.reduced_question.T)
        self.scores_inf=pd.DataFrame(data,index=range(len(data)))
        closest_docs_indices = self.scores_inf.sort_values(by=0, ascending=False).index[:self.top_n].values
        return closest_docs_indices,self.scores_inf

        

