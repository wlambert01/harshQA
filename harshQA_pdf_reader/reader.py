#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import re
import sys
import uuid
import time
import cProfile
import json
from itertools import repeat
import pandas as pd
import numpy as np
import nltk

""" If you never installed punkt and wordnet:
nltk.download('punkt') 
nltk.download('wordnet')
!python -m textblob.download_corpora
"""

from tqdm import tqdm
from tika import parser
from nltk import tokenize as tkn
from string import digits

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import words

from textblob import TextBlob
from pattern.en import conjugate, lemma, lexeme
from utils.utils import convert
pd.set_option('display.max_colwidth', 200)




class pdfconverter():
    """
    Pdf reader module that load all the pdf in your pdf directories and return a HarshQA dataframe with the pdf content
    
    args
    -------------------

    pdf_directories: list 
        list of path of directories, containing one or multiple pdfs files
    vocab_builder: str
        path to save the output .json vocabulary 
    save_directory: str
        path of the output directory that will contain saved .txt file (one file per pdf_directory) 
    save_names: list
        name of the output .txt files  (one file per pdf_directory) 
    """

    def __init__(self,pdf_directories,vocab_builder=None,save_directory=None,save_names=None):
        path=vocab_builder
        if path!=None:
            if  os.path.exists(path):
                with open(path) as json_data:
                    self.dic = json.load(json_data)  
        else:
            self.dic={}
            
        self.vocab=words.words()
        self.text_processor_pdf=np.vectorize(self.text_preprocessing_pdf,otypes=[str])
        self.cleaner=np.vectorize(self.extra_clean,otypes=[str])
        self.df = pd.DataFrame(columns=['pdf','directory','directory_index','raw_paragraphs','paragraphs'])
        self.parser=[]
        self.parser_raw=[]
        self.list_folder=[]
        self.paths={}
        self.save_directory=save_directory
        self.save_names=save_names
        self.vocab_builder=vocab_builder
        for dirs in pdf_directories:
            for r,d,f in os.walk(dirs):
               
                if d==[] and 'pdf' in '.'.join(f):
                    self.list_folder.append(r)

            for folder in self.list_folder:
                for i,pdf in enumerate(os.listdir(folder)):
                    if pdf!= '.DS_Store':
                        self.paths[folder]=self.paths.get(folder,[])+[(i,pdf)]
        
    def remove_iso_chars(self,text):
        chars=['\xa0','\xad']
        for token in chars:
            text=text.replace(token,' ')
        return text

        
        
    def check(self,word):
        word=word.lower()
        word=lemma(word)
        if self.dic.get(word,False)==1:
            return True
            
        else:
            boolean=word in self.vocab
            if boolean==True:
                self.dic[word]=1
            return boolean
        
    def check_multiprocessing(self,word):
        word=word.lower()
        word=lemma(word)
        if self.dic.get(word,False)==1:
            return True

        else:
            boolean=word in self.vocab
            if boolean==True:
                self.dic[word]=1
            return boolean
 
        
        
    
    def word_remerger(self,mot1,mot2):
        mot1,mot2=mot1.lower(),mot2.lower()
        if mot1.isdigit():
            if mot2.isdigit():
                return " ".join([mot1,mot2])
            else :
                return " ".join([mot1,mot2])
        else:
            if mot2.isdigit():
                return " ".join([mot1,mot2])
            else:
                merged="".join([mot1,mot2])
                if False in [self.check(mot1) , self.check(mot2), not self.check(merged)]:
                    return " ".join([mot1,mot2])
                else:
                    return merged
                
    def save_dic(self):
        with open(self.vocab_builder, 'w') as jsonfile:
            json.dump(self.dic, jsonfile)
        return self
    
    
    def update_dic(self,words):
        for w in words:
            self.dic[w]=1
        return self
        
    def transform(self):
        """Pdf-files reader with Apache Tika"""
        count=1
        assert len(self.list_folder)>=1 ,"FILES NOT FOUND"
        for i,folder in enumerate(self.list_folder):
            path=folder
            for j,pdf in enumerate(os.listdir(path)):
                
                
                if pdf!= '.DS_Store':
                    self.df.loc[count] = [pdf,folder.split('/')[-2], i+1,None,None]
                    
                    """ 0- Read Pdf file, remove isochars """
                    raw = parser.from_file(os.path.join(path,pdf))
                    s = raw['content']
                    s=self.remove_iso_chars(s)
                    
                    """ 1- Handle linebreaks to optimize TextBlob.sentences results"""
                    s=self.treat_new_line(s)
                    
                    """ 2- Divide text by sentences using TextBlob"""
                    blob=TextBlob(s)
                    paragraphs = np.array([str(s) for s in blob.sentences],dtype=str)
                    self.parser = []
                    self.parser_raw=[]
                    p=self.text_processor_pdf(paragraphs)
                    self.save_dic()
                    
                    
                    """
                    3- Get rid of bad text data:
                    Discard sentences with too long word (setted to n=20,  given the fact that 16 is the 99% percentile in terms of english words length)
                    Discard sentences with too much upper words (TITLE, CREDENTIALS, Link, ADDRESS .. )
                    """
                    
                    index_=[i for i,c in enumerate(self.parser) if (True in [len(w)>=20 for w in c.split()] )]
                    index_2=np.array(self.cleaner(self.parser),dtype=int)
                 
                    index_+=list(np.where(index_2==1)[0])
                    index_raw=[i for i,c in enumerate(self.parser_raw) if np.sum([ int(w[0]==w[0].upper())/len(c.split()) for w in c.split()])> 0.3]
                    index=list(set(index_ + index_raw))
                    self.df.loc[count,'paragraphs']=np.delete(np.array(self.parser),index)
                    self.df.loc[count,'raw_paragraphs']=np.delete(np.array(self.parser_raw),index)
                    count+=1
                            
            print("files from {} succesfully converted \n\n".format(folder))

        if self.save_directory!=None and self.save_names!=None:
            assert len(self.save_names)==self.df.directory_index.nunique(), "The nameslist to save pdfs to json must have same length than the number of repositories to read pdfs from"
            for i,name in enumerate(self.save_names):
                with open(self.save_directory+name+'.json', 'w') as outfile:
                    txt=list(self.df[self.df.directory_index==(i+1)]['paragraphs'])
                    txt=pd.Series(txt).to_json(orient='values')
                    json.dump(txt,outfile)


        return self.df
    
   
    
    def extra_clean(self,sentence):

        p=len(sentence.split())
        liste_words=sentence.split()
        multi_res=[self.check_multiprocessing(w) for w in liste_words]
        answ=False
        if (np.sum(multi_res)/p)<0.75:
            answ=True
        return int(answ)
        
        
    
    def remove_non_alpha(self,text):
        
        """ Remove non alpha-decimal caracters that are not dot or linebreaker """
        
        removelist="-\.\/\?\@"
        re_alpha_numeric1=r"[^0-9a-zA-Z"+removelist+" ]"
        clean_text=re.sub(re_alpha_numeric1,'',text)
        clean_text=clean_text.replace('/',' ')
        clean_text=re.sub(' +', ' ', clean_text)
        return clean_text
    
    def treat_new_line(self,text):
        """ 
        This function is aimed to deal with all types of linebreaks we met during our tests 
        There is linebreaks dure to cut-sentences, cut-words, bullet-list, title, new paragraphs, or sentences breaks
        """
        v1,v2='neutral','neutral'
        text=text.replace('.\n','. ')
        text=re.sub(r'(\n\s*)+\n+', '\n\n',text )
        
        lw=text.split('\n\n')
        lw=[c for c in lw if c.replace(' ','')!='']
            
        for i in range(1,len(lw)):
            

            el=lw[i]
            if len(el)>=1:
                try:
                    first_w=el.split()[0]
                except:
                    first_w=el
                first_l=first_w[0]
                if first_l.isupper() :
                    if len(lw[i-1])>0 and lw[i-1].replace(' ','') !='':
                        if lw[i-1].replace(' ','')[-1] not in [":",'.',"-",'/',"'",";"]:
                            prec=lw[i-1].split(".")[-1]
                            merge=(prec+' '+lw[i]).split()
                            dic=dict(nltk.tag.pos_tag(merge))
                            proper_noun=dic[first_w]=='NNP'
                            if not proper_noun:
                                if not "." in lw[i-1]:
                                    lw[i-1]=lw[i-1]+".\n\n "
                                else:
                                    lw[i-1]=lw[i-1][:-1]+".\n\n "
                            else:
                                lw[i-1]+=' '


                elif first_l.islower():
                    if len(lw[i-1])>0 and lw[i-1][-1].replace(' ','')!='':

                        if lw[i-1][-1].replace(' ','')[-1]!='-':
                            lw[i-1]+=""
                        else:

                            ltemp_prev=lw[i-1].split(' ')
                            ltemp_next=lw[i].split(' ')
                            motprev=ltemp_prev[-1][:-1]
                            motnext=lw[i].split(' ')[0]
                            if len((motprev+' '+motnext).split())==2:
                                newmot=self.word_remerger(motprev,motnext)
                                ltemp_prev[-1]=newmot
                                ltemp_next[0]=""
                                lw[i-1]=" ".join(ltemp_prev)
                                lw[i]=" ".join(ltemp_next)
                else:
                    lw[i-1]+="\n\n"
            
            
        text="".join(lw)
        
        lw=text.split('\n')
        lw=[c for c in lw if c.replace(' ','')!='']
        for i in range(1,len(lw)):
            #try:
            el=lw[i]
            if len(el)>=1:
                try:
                    first_w=el.split()[0]
                except:
                    first_w=el
                first_l=first_w[0]
                if first_l.isupper() :
                    if len(lw[i-1])>0 and lw[i-1].replace(' ','')!='':
                        if lw[i-1].replace(' ','')[-1] not in [":",'.',"-",'/',"'",";"]:
                            prec=lw[i-1].split(".")[-1]
                            merge=(prec+' '+lw[i]).split()
                            dic=dict(nltk.tag.pos_tag(merge))
                            proper_noun=dic[first_w]=='NNP'
                            if not proper_noun:
                                if not "." in lw[i-1]:
                                    lw[i-1]=lw[i-1]+".\n\n "
                                else:
                                    lw[i-1]=lw[i-1][:-1]+".\n\n "
                            else:
                                lw[i-1]+=' '
                elif first_l.islower():
                    if len(lw[i-1])>0 and lw[i-1].replace(' ','')!='':
                        if lw[i-1].replace(' ','')[-1]=="-":
                            ltemp_prev=lw[i-1].split(' ')
                            ltemp_next=lw[i].split(' ')
                            motprev=ltemp_prev[-1][:-1]
                            motnext=lw[i].split(' ')[0]
                            if len((motprev+' '+motnext).split())==2:
                                newmot=self.word_remerger(motprev,motnext)
                                ltemp_prev[-1]=newmot
                                ltemp_next[0]=""
                                lw[i-1]=" ".join(ltemp_prev)
                                lw[i]=" ".join(ltemp_next)
                        else:
                            lw[i-1]+=" "
                else:
                    lw[i-1]+=" "

        
        text="".join(lw)
        return text
        
    def cut_text(self,p):
        
        """ Cut text into sentences """
        undesirable_chars=['?','http','www','@']

        if (not True in [i in p for i in undesirable_chars]) and (len(p)>=70) and len(p.split())>=7 :
            
            phrases=self.remove_non_alpha(p)    
            phrases=phrases.replace('.',' ')
            phrases=phrases.replace('-',' ')
            phrases=phrases.replace("?"," ")
            phrases=re.sub(' +', ' ', phrases)
            phrases=re.sub(r'([0-9]+(?=[a-z])|(?<=[a-z])[0-9]+)',"",phrases)
            phrases=phrases.lower()
            self.parser.append(re.sub(' +', ' ', phrases))
            
        return None 
    
    def cut_text_raw(self,p):
        """Cut raw/untreated text into sentences """
        undesirable_chars=['?','http','www','@']
        if (not True in [i in p for i in undesirable_chars]) and (len(self.remove_non_alpha(p))>=70) and len(self.remove_non_alpha(p).split())>=7 :
            self.parser_raw.append(re.sub(' +', ' ', p))
            
        return None                                                                                                          
                                                                                                          
    def text_preprocessing_pdf(self,p):

        """ Pipeline of sentences-preprocessing using np.vectorize for faster results """

        cleaner=np.vectorize(self.remove_non_alpha,otypes=[str])
        cut_text=np.vectorize(self.cut_text,otypes=[str])
        cut_text_raw=np.vectorize(self.cut_text_raw,otypes=[str])
        assert len(self.parser)==len(self.parser_raw), "Length of the treated sentence treated list does not match length of raw text list: {} / {}".format(len(self.parser),len(self.parser_raw))
        cut_text_raw(p)
        p=cleaner(p)
        cut_text(p)
        return p



# In[1]:

"""
class voc():
    
    def __init__(self,dic):
        self.vocab=words.words()
        self.dic=dic
        
        
    def check(self,word):
        word=word.lower()
        word=lemma(word)
        if self.dic.get(word,False)==1:
            return True
            
        else:
            boolean=word in self.vocab
            if boolean==True:
                self.dic[word]=1
            return boolean
        
    def check_multiprocessing(self,word):
        l=[]
        word=word.lower()
        word=lemma(word)
        if self.dic.get(word,False)==1:
            return True,l

        else:
            boolean=word in self.vocab
            if boolean==True:
                l.append(word)
            return boolean,l
 
        
        
    
    def word_remerger(self,mot1,mot2):
        mot1,mot2=mot1.lower(),mot2.lower()
        if mot1.isdigit():
            if mot2.isdigit():
                return " ".join([mot1,mot2])
            else :
                return " ".join([mot1,mot2])
        else:
            if mot2.isdigit():
                return " ".join([mot1,mot2])
            else:
                merged="".join([mot1,mot2])
                if False in [self.check(mot1) , self.check(mot2), not self.check(merged)]:
                    return " ".join([mot1,mot2])
                else:
                    return merged
                
    def save_dic(self):
        with open('corpusESG.json', 'w') as jsonfile:
            json.dump(self.dic, jsonfile)
        return self
    def update_dic(self,words):
        for w in words:
            self.dic[w]=1
        return self
"""




