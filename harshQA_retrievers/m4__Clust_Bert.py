#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import m2__Bert
import m3__Tfidf
import json
import os
import re
import sys
import uuid
import prettytable
import time
import cProfile
import re
import pandas as pd
import numpy as np
import tinyarray
import nltk
import torch
from tqdm import tqdm
from tika import parser
from nltk import tokenize as tkn
from string import digits
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.feature_extraction.text as skf
from sklearn.base import BaseEstimator
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob
from models import InferSent
import enchant
from sentence_transformers import SentenceTransformer
from sklearn import decomposition
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import decomposition
from sklearn import datasets



# In[ ]:


class m4_Clust_BERT(BaseEstimator):
    """
    A scikit-learn estimator for TfidfRetriever. Trains a tf-idf matrix from a corpus
    of documents then finds the most N similar documents of a given input document by
    taking the dot product of the vectorized input document and the trained tf-idf matrix.
    
    Parameters
    ----------
    ngram_range : bool, optional
        [shape of ngram used to build vocab] (the default is False)
    max_df : bool, optional
        [while building vocab delete words that have a frequency>max_df] (the default is False)
    stop_words : str, optional
        ['english is the only value accepted'] (the default is False)
    paragraphs : iterable
        an iterable which yields either str, unicode or file objects
    top_n : int
        maximum number of top articles to retrieve
        header should be of format: title, paragraphs.
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
    >>> retriever = TfidfRetriever(ngram_range=(1, 2), max_df=0.85, stop_words='english')
    >>> retriever.fit(X=df['content'])
    
    >>> doc_index=int(input('Which document do you want to use for your question?'))
    >>> retriever.transform(X=df.loc[doc_index,'content'])
    
    >>> Q=str(input('Enter your question'))
    >>> Q=retriever.vectorizer.transform([Q])
    >>> closest_docs,scores = self.retriever.predict(newQst,df.loc[doc_index,'content'])
    """

    def __init__(self, TF_emb, q_emb, content, content_doc,stemmed_content, vocab, l_querries,l_querries_raw, by_querries=True, top_n=5,threshold_w=0.002, portion_w=1.0,rank=500,verbose=False):

        self.top_n = top_n
        self.verbose = verbose
        self.TF_emb=TF_emb
        self.threshold_importance=threshold_w
        self.prop_w=portion_w
        self.rk=rank
        self.by_querries=by_querries
        self.querries=l_querries.copy() #(dataframe querries with word importance)
        self.querries_raw=l_querries_raw.copy()
        self.content=content
        self.content_doc=content_doc
        self.stemmed_content=stemmed_content
        self.vocab=vocab
        self.dic_emb={}
        self.q_emb=q_emb
    
    def select_term_Naystrom(self,T):

        #T is the Tf-Idf Matrix 
        #s is the fraction of important words that we want to retrieve (s=1.0 generally)
        print('begin of term selection')
        #Retrieve all the stems of tf-idf vocabulary
        terms=self.vocab
        
        #First we remove digits of the terms candidate
        n=int(self.prop_w*len(terms))
        idx_words=[]
        for i,term in enumerate(terms):
            try: float(term)
            except: idx_words.append(i)

        #Secondly we set a threshold to select words that appear at least each p documents

        #Set threshold to select words that appear at least each p documents 
        self.freq_term=np.zeros((len(self.content),len(terms)))
        
        for j,stem in enumerate(self.stemmed_content):
            for i,c in enumerate(terms):
                if c in stem:
                    self.freq_term[j,i]+=1
                    
        self.freq_term_by_doc=np.apply_along_axis(lambda x: np.mean(x),0,self.freq_term)
        ids=np.where(self.freq_term_by_doc>self.threshold_importance )[0]
        idx_words=[ i for i in ids if i in idx_words]


        #Init selection of terms with most correlation 
        if self.prop_w!=1.0:
            prob=np.sum(np.abs(T)>0,axis=0)
            prob=prob/np.sum(prob)
            prob=np.squeeze(prob)
            idx=[]
            m=len(idx_words)

            for i in range(n):

                p=int(np.random.choice(m,1,prob[0]))
                idx.append(int(idx_words[p]))
                prob[idx_words[p]]=0
                prob=prob/np.sum(prob)

            return idx,np.squeeze(T[:,idx])
        
        else:
            idx=idx_words
            self.idx=idx
            return idx,np.squeeze(T[:,idx])
        print('end of term selection')
    
    

    def fit(self, X, y=None): 

        return self

    def transform(self): #generate new enhanced tf-idf-farahat features ( co-occurence weight frequency matrix) 
        
        idx,A=self.select_term_Naystrom(self.TF_emb.toarray())   
        
        #Full rank approximation
        if self.rk==-1:
            self.rk=len(idx)
                
        qemb=self.q_emb
        print('begin of generate kernel ')
        
        X=self.TF_emb.toarray().transpose()
        X=np.c_[X,qemb]

        L=np.eye(X.shape[0])*np.sqrt(X.shape[0])
        L_inv=np.linalg.inv(L)
        G=L_inv@X@X.transpose()@L_inv
        Gs=G[idx,:]
        Gs=Gs[:,idx]
        S,V,D=np.linalg.svd(Gs, full_matrices=True)
        Ssub,Vsub=S[:,:self.rk], np.diag(V)[:self.rk,:self.rk]
        #Ssub,Vsub=S,np.diag(V)
        #G=Ssub@Vsub$Ssub.transpose() but this operation is not necessary, we just need the decomposition of G

        D_sub_inv=np.diag(np.apply_along_axis(lambda x: 1/np.sqrt(x) , 0, V[:self.rk]))
        #D_sub_inv=np.diag(np.apply_along_axis(lambda x: 1/np.sqrt(x) , 0, V))
        W=(((D_sub_inv@Ssub.transpose())@X[idx,:])@(X.transpose()))@X
        self.TF_FARAHAT_emb=np.apply_along_axis(lambda x: x/np.sqrt(np.sum(x**2)), 1,W.transpose())
        print('end of generate kernel')
        return self

    
    

    

    def predict(self,metadata,repo):
        self.content_repo=self.content_doc[repo]
        questions=self.querries['query'].tolist()
        W_=self.TF_FARAHAT_emb
        scores=W_[-len(questions):]@(W_[metadata[0]:metadata[1]].transpose())
        rank=np.apply_along_axis(lambda x:np.argsort(-x),1,scores)
        raw_or_treated=1 #raw text
        
        all_answers=[]
        all_scores=[]
        all_indices=[]
        all_models=[]
        all_ranks=[]
        all_querries=[]
        bert=BertRetriever(top_n=5)
        for question in range(len(questions)):
            cluster_ids=[]
            answers=[]
            answers_raw=[]
            dic_answers={}
            count_answers=1
            #print('*****  Question {} : {} {}'.format(question,questions[question],'********'))
            rk=0
            count=0
            #print('\n----------------\n')
            w_important=[w for w in questions[question].split(" ") if w==w.upper() ]
            #Display best candidates according to clustering
            
            ### Special words retrieval (Tf-Idf search) ###
            if w_important !=[]:
          
                for w in w_important:
                    ll=m3_Tfidf().tokenize(w)
                    #print("w,ll=",w,ll)
                    if ll!=[]:
                        w=ll[0]
                        best_id=[i for i in range(rank.shape[1]) if w in self.content_repo[0][rank[question,i]].split(' ')]
                        if best_id!=[]:
                            id_match=best_id[0]
                            result=self.content_repo[raw_or_treated][rank[question,id_match]]
                            #print('Special Anserw : {}'.format(self.content_repo[raw_or_treated][rank[question,id_match]]))
                            #print('\n----------------\n')


                            all_indices.extend([rank[question,id_match]])
                            all_scores.extend([scores[question,rank[question,id_match]]])
                            all_answers.extend([result])
                            all_ranks.append(rk+1)
                            all_models.append("Tf-Idf- Semantic Kernel- Bert")
                            all_querries.append(self.querries_raw[question])
                            rk+=1
                            count_answers+=1
            
            ###Normal Loop Spherical Kmeans Clustering and Applying Bert encoder###
            #1 Get clusters
            #print('begin retriever from cluster')
            for answ in range(size_cluster):
                while True:
                    answer_raw=self.content_repo[raw_or_treated][rank[question,count]]
                    answer=self.content_repo[1-raw_or_treated][rank[question,count]]
                    dic_answers[answer]=dic_answers.get(answer,0)+1
                    if dic_answers[answer]==1:
                        break
                    cluster_ids.append(metadata[0]+rank[question,count])
                    count+=1
                
                answers.append(answer)
                answers_raw.append(answer_raw)

                #print('Anserw n°{} : {}'.format(count_answers,answer))
                #print('\n----------------\n')
                
            #print('end loop cluster retriever')

            #2) Use Bert encoder inside clusters + cosine similarity retrieval
            Qst=questions[question].lower()
            newQst=pdfconverter().remove_non_alpha(Qst)
            newQst=newQst.replace('.','')

            #print('initiate bert cluster retriever')
            bert.embeddings=[]
            bert.reduced_embeddings=[]
            
            
            #embeddings = bert.transform(list([s for s in answers ]))
            try:
                bert.embeddings = [QAmodel.bert.embeddings[i] for i in cluster_ids]
                bert.reduced_embeddings = np.apply_along_axis(lambda v: bert.normalize(v), 1,bert.embeddings)
            except:
                indices_saved=[i for (i,c) in enumerate(answers) if c in self.dic_emb.keys()]
                print('just saved {} bert encodings'.format(len(indices_saved)))
                indices_new=[i for i in range(len(answers)) if i not in indices_saved ]
                indices_sort=indices_saved+indices_new
                indices_unsort=np.argsort(indices_sort)
                saved_embeddings=np.array([self.dic_emb[answers[i]] for i in indices_saved])
                new_item=[answers[i] for i in indices_new]
                if indices_new!=[]:
                    bert.transform(list(new_item))
                    new_embeddings=bert.reduced_embeddings
                
                for (i,s) in enumerate(new_item):
                    self.dic_emb[s]=new_embeddings[i]
                    
                if indices_saved!=[]:
                    if indices_new==[]:
                        bert.reduced_embeddings=saved_embeddings
                    else:
                        bert.reduced_embeddings=np.r_[saved_embeddings,new_embeddings][indices_unsort]
                
                
                
            
            indices,scores_bert=bert.predict(newQst,[0,len(answers)])
            scores_bert=scores_bert.loc[indices].values[:,0]
            text=[ answers_raw[i] for i in indices]
            for i,c in enumerate(text):
                
                #print('Anserws n° {} : {}'.format(i+1,c))
                #print('\n----------------\n')
                all_ranks.append(rk+1)
                all_models.append("Tf-Idf- Semantic Kernel- Bert")
                all_querries.append(self.querries_raw[question])
                rk+=1
               
            all_answers.extend(text)
            all_scores.extend(scores_bert)
            all_indices.extend(rank[question,:size_cluster][indices])
            #all_indices.extend(np.array(range(len(self.content_repo[0])))[rank[question,:self.top_n]])
            
        return pd.DataFrame(np.c_[all_querries,all_models,all_ranks,all_indices,all_answers, all_scores],columns=['Question','Model','Rank','Doc_index','Answer','Score'])
    
    

