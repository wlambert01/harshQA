#!/usr/bin/env python
# coding: utf-8

# In[4]:

import warnings
import os
import re
import sys
import numpy as np
import pandas as pd
import nltk
from tqdm import tqdm
from nltk import tokenize as tkn
from harshQA_retrievers.m1__Infersent import m1_Infersent

def remove_non_alpha(text):
    
    """ Remove non alpha-decimal caracters that are not dot or linebreaker """
    
    removelist=""
    re_alpha_numeric1=r"[^0-9a-zA-Z"+removelist+" ]"
    clean_text=re.sub(re_alpha_numeric1,'',text)
    clean_text=clean_text.replace('/',' ')
    clean_text=re.sub(' +', ' ', clean_text)
    return clean_text

def generate_querries(querries,w_path,m_path,stopwords,importance=0.75):
        
        inferst=m1_Infersent(w2v_path=w_path,model_path=m_path)
        querries=[remove_non_alpha(q) for q in querries]    
        sentences=[s[0].lower()+s[1:].replace('?','')  for s in querries]
        i=0
        important_words=[]
        unsorted_words=[]

        for qu in sentences:

            #Get scores from infersent visualization function (max-pooling et each layer)
            tensor,vector,scores,words=inferst.infersent.visualize(qu)
            scores=np.array(scores[1:len(scores)-1])
            words=np.array(words[1:len(words)-1])

            #Remove stopwords from querries and attributed scores
            pos=[i for i,c in enumerate(list(words)) if c not in stopwords]
            words=words[pos]
            scores=np.array(scores)[pos]
            scores=scores/np.sum(scores)

            #Sort query words by word importance keeping idx in memory to unsort it back
            data=pd.DataFrame(np.c_[words,scores],columns=['word','score'])
            idx=np.argsort(-np.array(data.score.values,dtype='float64'))
            
            data=data.sort_values(by=['score'],ascending=False)
            new_words=data.word.values
            new_scores=np.array(data.score.values,dtype=float)

            #Keeping a set of words that satisfy 75% of cumulative importance
            score_cum=np.cumsum(new_scores)
            pos=np.where(score_cum<importance)[0]
            

            lw=new_words[pos]
            ls=new_scores[pos]
            idx_unsort=np.argsort(idx[pos])
            lw_unsort=lw[idx_unsort]

            important_words.append(lw)
            unsorted_words.append(lw_unsort)
            i+=1
        
        array=np.zeros((len(querries),3),dtype=object)
        for i,c in enumerate(querries):
            array[i]=np.array([c,important_words[i],unsorted_words[i]])
            
        return pd.DataFrame(array,columns=['query','words_sort','words_unsort'])

        """
        idx=[5,2,0,3,1,4]
        idx°unsort=[2,4,1,3,5,0]
        pos=[0,1,2]
        """


def hide_warn(*args, **kwargs):
    pass

def convert(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError

