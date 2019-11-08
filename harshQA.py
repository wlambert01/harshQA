#!/usr/bin/env python
# coding: utf-8

## ESG Risk Assessment Projects BNP
## Pipeline of code for question answering on closed domain and non factoid questions - harshQA :)
## Developped by William Lambert (Risk AIR Team , BNP Paribas) , contact : william.lambert@ensae-paristech.fr
## ESG Project: Risk ESG (Marie Simon), Risk AIR (Lea Deleris, William Lambert)

import warnings
from utils.utils  import hide_warn
warnings.warn=hide_warn
import json
import os
import re
import sys
import uuid
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf 
from sentence_transformers import SentenceTransformer
from string import digits
from sklearn.base import BaseEstimator
from tabulate import tabulate

#Bert_dependencies
from harshQA_reranker.tokenization import* 
import harshQA_reranker.metrics as metrics
import harshQA_reranker.modeling as modeling
import harshQA_reranker.optimization as optimization 

#Import our pdf reader
from harshQA_pdf_reader.reader import pdfconverter

#Import bert finetuned pipeline
from harshQA_reranker.harshQA_tfrecord import *
from harshQA_reranker.harshQA_bert_builder import * 
from harshQA_reranker.harshQA_run_msmarco import run_msmarco

#Import all our models, wrap in a scikit learn estimator
from harshQA_retrievers.m1__Infersent import m1_Infersent
from harshQA_retrievers.m2__Bert import m2_Bert
from harshQA_retrievers.m3__Tfidf import m3_Tfidf
from harshQA_retrievers.m5__harshQA import m5_harshQA

#Utils
from utils.utils import remove_non_alpha
from utils.utils import generate_querries


# ## Flags

# In[2]:


flags = tf.flags
FLAGS = flags.FLAGS


# MODEL SELECTION 
flags.DEFINE_integer(
    "model",5,
    "** Select a model \n **Model without pre-clustering: \n\t 1-Infersent_glove [Pretrained] (10 min/corpus) \n\t 2-Bert [Pretrained on SQUAD] (30 min/corpus) \n \t 3-Tf_Idf_Lemmatizer [Trained on our corpus] (5 min/ corpus) \n**Model with pre-clustering\\t 4-Tf-Idf_Bert [Pretrained on SQuAD] (3 min/query)\n\t 5-Tf-Idf_Bert_enhanced [Finetuned on MsMarco] (1:30 min/query)\n\t 6- All \n")
"""
*** 
** Model without pre-clustering:
        1-Infersent_glove [Pretrained] (10 min/corpus)
        2-Bert [Pretrained on SQUAD] (30 min/corpus)
        3-Tf_Idf_Lemmatizer [Trained on our corpus] (5 min/ corpus)
        
* *Model with pre-clustering
        4-Tf-Idf_Bert [Pretrained on SQuAD] (3 min/query)
        5-Short text clustering (enhanced tf-idf) and Bert retriever finetuned on MsMarco (1:30 min/query)


** The settings of our test was: *** 
  Run on CPU
  size_cluster=50
  Corpus of text was 1500 sentences(300 pages) and 15 queries
  Corpus of domain_vocab was 3000 sentences (600 pages) 
  The timespeed of pdf converter is approximately 10s/1000 pages\
  The best results were achieved with model5 harshQA ; see our notebook : harshQA_eval

"""
flags.DEFINE_boolean("demo",True,"")
flags.DEFINE_string("demo_query","Does the company reduce its ghg emissions?","")
flags.DEFINE_string("demo_topics","reduce emissions, ghg emissions","")
flags.DEFINE_integer("top_n",5,"")
flags.DEFINE_string("query_dir",'./utils/pdf_files/Tourism/Queries.txt',"")
flags.DEFINE_integer("size_cluster",100,"")
flags.DEFINE_string("domain", "Tourism" ,"")
flags.DEFINE_string("retrieved_company","Disney","")
flags.DEFINE_string("pdf_directory",'./utils/pdf_files/',"")
flags.DEFINE_string("vocab_file","./data/bert/pretrained_models/uncased_L-12_H-768_A-12/vocab.txt","")
flags.DEFINE_string("whole_corpus", "./utils/pdf_files/All","")
flags.DEFINE_string("vocab_builder","./data/corpusESG.json","")
flags.DEFINE_string("w2v_path", "./data/fastText/crawl-300d-2M.vec","")
flags.DEFINE_string("model_path","./data/encoder/infersent2.pkl","")    
flags.DEFINE_string("output_dir","./data/output","")
flags.DEFINE_string("bert_config_file", "./data/bert_msmarco/bert_config.json","")
flags.DEFINE_string("init_checkpoint","./data/bert_msmarco/model.ckpt","")
flags.DEFINE_integer("max_seq_length", 512,"")
flags.DEFINE_integer("max_query_length", 128,"")
flags.DEFINE_boolean("msmarco_output", True,"")
flags.DEFINE_bool("do_train", False, "")
flags.DEFINE_bool("do_eval", True, "")
flags.DEFINE_integer("train_batch_size", 20, "")
flags.DEFINE_integer("eval_batch_size", 20, "")
flags.DEFINE_float("learning_rate", 1e-6, "")
flags.DEFINE_integer("num_train_steps", 400000,"")
flags.DEFINE_integer("max_eval_examples", None,"")
flags.DEFINE_integer("num_warmup_steps", 40000,"")
flags.DEFINE_integer("save_checkpoints_steps", 100,"")
flags.DEFINE_integer("iterations_per_loop", 10,"")
flags.DEFINE_integer("min_gram",1,"")
flags.DEFINE_integer("max_gram",1,"")
flags.DEFINE_bool("lemmatize",False,"")
flags.DEFINE_bool("transform_text",True,"")
flags.DEFINE_integer("sentences_chunk",1,"")
flags.DEFINE_bool("use_tpu", False, "")
tf.flags.DEFINE_string("tpu_name", None,"")
tf.flags.DEFINE_string("tpu_zone", None,"")
tf.flags.DEFINE_string("gcp_project", None, "")
tf.flags.DEFINE_string("master", None, "")
flags.DEFINE_integer("num_tpu_cores", 8,"")



############ ARGS CHECK #################

def check_args():
    
    #Checking tensorflow version
    assert tf.__version__[0]=='1' , "This code has been implemented on tf 1.14, if you want to use you should modify the code, or consider to downgrade your tf version"

    #Checking if a company has been properly selected
    assert FLAGS.retrieved_company!=None, "Select a Company"
    assert FLAGS.domain!=None, "Select a Domain"
    assert FLAGS.model!=None, "Select a Model"

    #Forcing the architecture of pdfs data 
    assert FLAGS.pdf_directory!=None, "Select a $PATH which contains the folder Domain/Company/pdfs/ containing pdf files to query"
    assert os.path.isdir(FLAGS.pdf_directory+FLAGS.domain), "Your pdf directory must contains a $FLAGS.domain folder "
    assert os.path.isdir(FLAGS.pdf_directory+FLAGS.domain+'/'+FLAGS.retrieved_company), "Your FLAGS.pdf_directory/FLAGS.domain/ must contain a $FLAGS.retrieved_company folder"
    assert os.path.isdir(FLAGS.pdf_directory+FLAGS.domain+'/'+FLAGS.retrieved_company+'/'+'pdfs'), "Your FLAGS.pdf_directory/FLAGS.domain/FLAGS.retrieved_company must include a pdfs/ folder that contains the pdf files"

    #Forcing the architecture of output_data
    assert os.path.isdir(FLAGS.output_dir+"/"+"tf_idf_checkpoints"), "output_dir must contain a tf_idf_checkpoints folder to save idfs , kernel and tf-idf vocab. The folder can be empty. "
    assert os.path.isdir(FLAGS.output_dir+"/"+"bert_checkpoints"), "output_dir must contain a bert_checkpoints folder. The folder can be empty  "
    assert os.path.isdir(FLAGS.output_dir+"/"+"tfrecord"), "output_dir must contain a tfrecord folder. The folder can be empty"

    #Checking other args
    assert FLAGS.whole_corpus!=None, "Select a pdf directory path which contains all the pdf of the corpus to fit our model"
    assert FLAGS.vocab_builder!=None, "Enter a .json path to save all our vocabulary while ingesting pdf files "
    assert FLAGS.lemmatize==False or FLAGS.model==3 ,"Lemmatize option is only available for Tf-Idf model (model n°3)"
    assert FLAGS.max_gram==1 or FLAGS.model==3 , "Multi gram option is only available for Tf-Idf model (model n°3)"
    assert FLAGS.transform_text==True, "Your model will work better with a tokenizer that used (stemming/lemmatizing)"
    
    #Args for harshQA model
    if FLAGS.model ==5:
        assert FLAGS.eval_batch_size<=FLAGS.size_cluster,"eval batch size should be less than the size of the cluster of preselected sentences"
        assert FLAGS.size_cluster%FLAGS.eval_batch_size==0,"eval batch size should be a multiple of the size of the cluster of preselected sentences"
        assert FLAGS.bert_config_file!=None, "Enter a .json bert config file to specify the model architecture"
        assert FLAGS.vocab_file!=None, "Enter the path of uncased_L-12_H-768_A-12/vocab.txt"
        assert "uncased_L-12_H-768_A-12" in FLAGS.vocab_file, "You need to pass the vocab file of bert uncased L-12_H-768_A-12 "
        assert ".txt" in FLAGS.vocab_file, "The bert vocab_file must be a .txt file"
        assert ".json" in FLAGS.bert_config_file, "The bert_config_file must be a .json file"
        assert FLAGS.output_dir!=None, "Enter the output directory where all the model bert,tfidf checkpoints will be written after train "
        "It will also store the raw tsv files and the tfrecords used to feed bert-reranker."
        assert FLAGS.init_checkpoint!=None,"Enter a bert .ckpt init checkpoint"
        assert ".ckpt" in FLAGS.init_checkpoint, "The init_checkpoint must be a .ckpt file"
        
    
    #Args for harshQA and Infersent models
    if FLAGS.model in [1,5]:
        assert FLAGS.w2v_path!=None,"Specify the .vec file path of GloVe or fasText"
        assert ".vec" in FLAGS.w2v_path, "The w2v path of GloVe or fasText muste be a .vec file"
        assert FLAGS.model_path!=None,"Specify the .pkl file path of Infersent2"
        assert ".pkl" in FLAGS.model_path, "The infersent model file muste be a .pkl file"
            
    #Args for demo mode 
    if FLAGS.demo:
        assert FLAGS.demo_query!=None, "Specify a query for the demo"
        if FLAGS.model==5:
            assert FLAGS.demo_topics!=None, "Specify coma separated topics linked to your query for the demo"
    else:
        assert FLAGS.query_dir!=None , 'Specify a .txt file containing your queries line by line'

        

###### Question Answering Pipeline ##############

class QApipeline():

    """
    QAPipeline for closed domain question answering for non factoid questions, built on a scikit learn wrapper.
    User can select a model within the list provided below, and predict multiple answers for a set of queries.

    ### Args : 


    BASIS ARGS for all models: 

        # model : Choose a models between all of this :

            Model without pre-clustering (transform the whole corpus):
                1-Infersent_glove [Pretrained] (slow but quite accurate)
                2-Bert [Pretrained on SQUAD] (really slow on big corpus but good accuracy)
                3-Tf_Idf_Lemmatizer [Trained on our corpus]  (fast but less accurate)
            Model with pre-clustering
                5- harshQA model : Short text clustering + Query Expansion + Bert retriever finetuned on MsMarco  (medium speed and high accuracy)

        # demo : "Demo mode with your own pdfs."
        # demo_query : "[If Demo=True] Query ."
        # demo_topics : "[If Demo=True] Topics (coma separated)"
        # query_dir : " [If Demo=False] Path of the .txt file where your queries are located"

        # domain : "Domain folder name to process Q&A."
        # retrieved_company : "Company folder name to query."
        # pdf_directory :  "Path of the pdf directory."
        # whole_corpus: "The whole corpus to fit before predicting, smaller corpus, i.e the corpus"
                    "to build and save the tfidfs weights, vocab and semantic kernel."
        # vocab_builder : "The path to build our own vocabulary file for the pdf-reader module: will speed up reading after few pdfs ! "
        # top_n : "Number of doc to retrieve per query"
        # sentences_chunk : "Whether to use block of one or two sentences (min=1 , max=2) as answers"
                         "Keep in mind that we return the additional context of each sentences at the end of the prediction loop,"
                        "i.e the previous and next sentence of the answer."

    MODEL DEPENDENT ARGS: 

        REQUIRED ARGS without default values :

            # vocab_file : [If model=5 only] "The vocabulary file that the BERT model was trained on."
           
            # w2v_path : [If model in [1,5] only] "Path of the .vec file of GloVe or FastText."
            # model_path : [If model in [1,5] only] "Path of the .pkl file of infersent model."
            # output_dir : [If model=5 only] "The output directory where all the model bert,tfidf checkpoints will be written after train "
                    "Will also store the raw tsv and tfrecords predictions."
            # bert_config_file : [If model=5 only] "This specifies the model architecture."
            # init_checkpoint : [If model=5 only]"Initial checkpoint (usually from a pre-trained BERT model)."

            # size_cluster : "[If model=5 only] Size of the clusters of candidate to feed in neural network."

        REQUIRED ARGS with default values:

            # lemmatize : [If model=3 only else keep it False] "Whether to use lemmas instead of stems (not advised at all)."
            # transform_text : [If model in [3,5] only] "Whether to use transform text or not (stemmning by default).
            # min_gram : [If model=3] "Min grams to use for tf-idf model retriever."
            # max_gram : [If model=3] "Max grams to use for tf-idf model retriever."

            # max_seq_length : [If model=5 only] "The maximum total input sequence length after WordPiece tokenization. "
                    "Sequences longer than this will be truncated, and sequences shorter"
                    "than this will be padded."
            # max_query_length : [If model=5 only] "The maximum query sequence length after WordPiece tokenization. "
                    "Sequences longer than this will be truncated."
            # msmarco_output : [If model=5 only] "Whether to write the predictions to a MS-MARCO-formatted file."
            # do_train : [If model=5 only] "Whether to run training."
            # do_eval : [If model=5 only] "Whether to predict or not."
            # train_batch_size : [If model=5 only] "Total batch size for training."
            # eval_batch_size : [If model=5 only] "Total batch size for eval."
            # learning_rate: [If model=5 only] "The initial learning rate for Adam.""
            # num_train_steps : [If model=5 only] "Total number of training steps to perform."
            # max_eval_examples : [If model=5 only] "Maximum number of examples to be evaluated."
            # num_warmup_steps :  [If model=5 only] "Number of training steps to perform linear learning rate warmup."
            # save_checkpoints_steps : [If model=5 only] "How often to save the model checkpoint."
            # iterations_per_loop: [If model=5 only] "How many steps to make in each estimator call."

        OPTIIONAL ARGS : 

            # use_tpu : [If model=5 only] "Whether to use TPU or GPU/CPU."
            # tpu_name : [If model=5 only] "The Cloud TPU to use for training. This should be either the name used when creating 
                    the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url."
            # tpu_zone [If model=5 only] "GCE zone where the Cloud TPU is located in. If not specified, we will attempt 
                    to automatically detect the GCE project from  metadata.
            # gcp_project : [If model=5 only]  "Project name for the Cloud TPU-enabled project. If not specified, we will attempt to automatically 
                    detect the GCE project from metadata "
            # master : [If model=5 only] " TensorFlow master URL."
                    num_tpu_cores : "Only used if `use_tpu` is True. Total number of TPU cores to use."


    ### Examples :

    harshQA in demo mode: 

   python harshQA.py \
      --demo=True\
      --model=5\
      --domain="Tourism"\
      --retrieved_company="Disney"\
      --demo_query="Does the company support local agriculture?"\
      --demo_topics="local agriculture , support agriculture"\
      --top_n=5\
      --transform_text=True\
      --size_cluster=50\
      --eval_batch_size=50\
      --pdf_directory="./utils/pdf_files/"\
      --whole_corpus="./utils/pdf_files/All"\
      --vocab_file="./data/bert/pretrained_models/uncased_L-12_H-768_A-12/vocab.txt"\
      --vocab_builder="./corpusESG.json"\
      --w2v_path="./data/fastText/crawl-300d-2M.vec"\
      --model_path="./data/encoder/infersent2.pkl"\
      --init_checkpoint="./data/bert_msmarco/model.ckpt"\
      --bert_config_file="./data/bert_msmarco/bert_config.json"\
      --output_dir="./output"

    harshQA in auto mode :
   python harshQA.py \
      --demo=True\
      --model=5\
      --domain="Chemicals"\
      --retrieved_company="Pfizer"\
      --query_dir="./utils/pdf_files/Tourism/Queries.txt"\
      --top_n=5\
      --transform_text=True\
      --size_cluster=50\
      --eval_batch_size=50\
      --pdf_directory="./utils/pdf_files/"\
      --whole_corpus="./utils/pdf_files/All"\
      --vocab_file="./data/bert/pretrained_models/uncased_L-12_H-768_A-12/vocab.txt"\
      --vocab_builder="./corpusESG.json"\
      --w2v_path="./data/fastText/crawl-300d-2M.vec"\
      --model_path="./data/encoder/infersent2.pkl"\
      --init_checkpoint="./data/bert_msmarco/model.ckpt"\
      --bert_config_file="./data/bert_msmarco/bert_config.json"\
      --output_dir="./output"


    Any other models auto mode (here an example with Tf-Idf, i.e model n°3):

    # Demo mode: 

    python harshQA.py \
      --demo=True\
      --model=3\
      --domain="Tourism"\
      --demo_query="Does the company reduce its carbon emissions?"\
      --retrieved_company="Disney"\
      --top_n=5\
      --transform_text=True\
      --pdf_directory="./utils/pdf_files/"\
      --whole_corpus=".utils/pdf_files/All/"\
      --vocab_builder="./corpusESG.json"\
      --output_dir="./output"



    # Auto mode :

    python harshQA.py \
      --demo=False\
      --model=3\
      --domain="Chemicals"\
      --query_dir="./utils/pdf_files/Tourism/Queries.txt"\
      --retrieved_company="Novartis"\
      --top_n=5\
      --transform_text=True\
      --pdf_directory=".pdf_files/"\
      --whole_corpus=".pdf_files/All/"\
      --vocab_builder="./corpusESG.json"\
      --output_dir="./output"
      
    """

    
    def __init__(self,**kwargs):
        
        
        #new kwargs: 'threshold' (float between 0.5 and 1.0)
        
        self.kwargs_converter = {key: value for key, value in kwargs.items()
                            if key in pdfconverter.__init__.__code__.co_varnames}
        
        self.kwargs_Tf_Idf = {key: value for key, value in kwargs.items()
                         if key in m3_Tfidf.__init__.__code__.co_varnames}
        
        self.kwargs_Infersent={key: value for key, value in kwargs.items()
                         if key in m1_Infersent.__init__.__code__.co_varnames}
        self.kwargs_Bert={key: value for key, value in kwargs.items()
                         if key in m2_Bert.__init__.__code__.co_varnames}
        

        self.MODELS=['INFERSENT - GLOVE','BERT PRETRAINED','TFIDF - LEMMATIZER & BIGRAM','BERT & TFIDF SHORT TEXT CLUSTERING','BERT FINETUNED & TFIDF SHORT TEXT CLUSTERING']
        self.usemodel=FLAGS.model
        self.sentences_chunk=kwargs['sentences_chunk']
        return None


    def fit_reader(self,df=None):
        
        #CALL PDF READER 
        if df!=None:
            assert not False in [ col in df.columns.tolist() for col in ['directory_index','raw_paragraphs','paragraphs']], "The given dataframe is not of the proper format "
            self.df=df
        else:
            try:
                print('{}*********** READER *************'.format('\n'))
                print("Reading pdfs doc on location: {}".format(FLAGS.pdf_directory+FLAGS.domain+'/'+FLAGS.retrieved_company+'/pdfs/'))
                self.df=pdfconverter(**self.kwargs_converter).transform()

            except:
                print('You should give either your own harshQA dataframe to the fit_reader module or specify your pdf_directories, domain and retrieved_company FLAGS')


                
        #BUILD CONTENT AND DOCUMENT INDEX 
        self.content=[]
        self.content_raw=[]
        self.contents_doc=[]
        self.borders=[0]

        print('********* DOCUMENTS RETRIEVED **********')
        for j,repo in enumerate(sorted(list(set(self.df.directory_index)))):
            
            count_dic=[{},{}]
            remove_idx=[[],[]]
            content_doc=[]
            content_doc_raw=[]
            
            title=self.df[self.df.directory_index==repo].directory.tolist()[0]
            self.df[self.df.directory_index==repo]['raw_paragraphs'].apply(lambda sentences: self.update_count_dic(sentences,count_dic,0,remove_idx))
            self.df[self.df.directory_index==repo]['paragraphs'].apply(lambda sentences: self.update_count_dic(sentences,count_dic,1,remove_idx))
            self.df[self.df.directory_index==repo]['raw_paragraphs'].apply(lambda sentences: content_doc_raw.extend(sentences))
            self.df[self.df.directory_index==repo]['paragraphs'].apply(lambda sentences: content_doc.extend(sentences))
            
            #REMOVE TWIN SENTENCES AND REMOVE TOO SMALL SENTENCES
            remove_idx=list(set(remove_idx[0]+remove_idx[1]))
            content_doc=np.delete(np.array(content_doc),remove_idx)
            content_doc_raw=np.delete(np.array(content_doc_raw),remove_idx)
            
            content=[content_doc[i] for i in range(len(content_doc)) if (len(content_doc[i])>=50 )]
            content_raw=[content_doc_raw[i] for i in range(len(content_doc)) if (len(content_doc[i])>=50)]
            self.borders.append(len(content))
            
            print("FOLDER : {} , {} sentences \n \n".format(self.df.directory_index.unique()[j],len(content)))
            
            #ADD SENTENCES TO OUR FINAL OBJECTS
            self.content.extend(list(content))
            self.content_raw.extend(list(content_raw))
            self.contents_doc.append([content,content_raw])
        
        
        self.borders=list(np.cumsum(self.borders))
        
        #GROUP SENTENCES BY PAIR EVENTUALLY
        if self.sentences_chunk==2:

            self.content=[ ' '.join(x) for x in zip(self.content[0::2], self.content[1::2]) ]
            self.content_raw=[ ' '.join(x) for x in zip(self.content_raw[0::2], self.content_raw[1::2]) ]
            for i,(treated_sentences,raw_sentences) in enumerate(self.contents_doc):
                self.contents_doc[i][0]=[ ' '.join(x) for x in zip(treated_sentences[0::2], treated_sentences[1::2]) ]
                self.contents_doc[i][1]=[ ' '.join(x) for x in zip(raw_sentences[0::2], raw_sentences[1::2]) ]
            self.borders=[int(i/2) for i in self.borders]

        #REPLACE ALL DIGITS WITH SPECIAL TOKEN FOR OUR MODEL
        for i,c in enumerate(self.borders[:-1]):
            start_idx=self.borders[i]
            content=self.contents_doc[i][0]
            content_raw=self.contents_doc[i][1]

            #ADD TREATED TEXT TO CONTENTS_DOC
            for sentence_id,sentence in enumerate(content):
                
                words_list=sentence.split(" ")
                for word_id,w in enumerate(words_list):
                    try:
                        float(w)
                        words_list[word_id]="XXX"
                    except:
                        words_list[word_id]=w
                    self.content[start_idx+sentence_id]=" ".join(words_list)
                    
                self.contents_doc[i][0][sentence_id]=" ".join(words_list)

        return self
    
    def fit(self):
        
        
        print('********* MODEL {} **********'.format(self.MODELS[self.usemodel-1]))

        if self.usemodel==1:
            #Fit Infersent
            self.model_retriever = m1_Infersent(**self.kwargs_Infersent)
            self.model_retriever.fit(self.content)
            self.model_retriever.transform(self.contents_doc[0][0])
        
        if self.usemodel==2:
            #Fit Bert pretrained
      
            self.model_retriever=m2_Bert(**self.kwargs_Bert)
            self.model_retriever.fit(self.content)#no finetuning for Bert
            self.model_retriever.transform(self.contents_doc[0][0])
           
        if self.usemodel==3:
            #Fit Tf-Idf model

            """
            This model is also called by HarshQA model (m5), so we need to turn off the saving output feature so that 
            it does not erase the harshQA output.
            """
            self.kwargs_Tf_Idf['save_idfs_path']=None
            self.kwargs_Tf_Idf['save_features_path']=None


            self.model_retriever = m3_Tfidf(**self.kwargs_Tf_Idf)
            self.model_retriever.fit(self.content)
            self.model_retriever.transform(self.contents_doc[0][0])
 
        
        
        if self.usemodel==5:
            #Fit harshQA model 
           
            output_TF=FLAGS.output_dir+'/tf_idf_checkpoints/'
            
            args_harshQA={'save_kernel_path':output_TF+'kernel.npy',
                          'save_kernel_vocab_path':output_TF+'kernel_vocab.json',
                          'save_kernel_idx_path':output_TF+'kernel_vocab_idx.json',
                          'save_idfs_path':output_TF+'idfs.npy',
                          'save_features_path':output_TF+'vocab.json'}
            
            for key, value in FLAGS.flag_values_dict().items():
                if key in m5_harshQA.__init__.__code__.co_varnames:
                    args_harshQA[key]=value
                    
            self.model_retriever=m5_harshQA(**args_harshQA)
            self.model_retriever.fit(self.content)
            self.model_retriever.transform(self.contents_doc[0])
            
        return self
    
    
        #Initialisation of Tf-Idf-Farahat
    
    def predict(self,Qst,Topics=None):
        """
        kwargs:
        ##VE_type: 'DP' for Detect Presence of 'VE' for Value extraction
        ##Qst: Querry
        ##VE_cdt : null
        """
        
        repo_to_query=0
        self.Qst_raw=Qst
        self.topics=Topics

        #Apply corpus transformations to querry before feeding it into our models
        newQst=[q.lower() for q in self.Qst_raw]
        newQst=[remove_non_alpha(q) for q in newQst]
        newQst=[q.replace('.','') for q in newQst]

        self.dataframe=[]        
        all_scores=[]
        all_models=[]
        all_querries=[]
        all_ranks=[]
        all_indices=[]
        all_answers=[]

        #Infersent retriever
        if self.usemodel !=5:
            for i,qu in enumerate(newQst):

                indices,scores=self.model_retriever.predict(qu)
                p=len(indices)
                all_scores.extend(scores.loc[indices].values[:,0])
                all_answers.extend([ self.contents_doc[repo_to_query][1][i] for i in indices])
                all_models.extend([self.MODELS[FLAGS.model-1]]*p)
                all_ranks.extend(list(range(1,p+1)))
                all_querries.extend([self.Qst_raw[i]]*p)
                all_indices.extend(indices)

           
            self.dataframe=pd.DataFrame(np.c_[all_querries,all_models,all_ranks,all_indices,all_answers,all_scores],columns=['Question','Model','Rank','Doc_index','Answer','Score'])
                 
        
        #harshQA retriever
        else:
            self.dataframe=self.model_retriever.predict(self.Qst_raw,self.topics)
      

        #FORMAT THE OUTPUT NICELY AND RETURN IT
        self.dataframe=self.dataframe.apply(self.add_ctxt,axis=1)
        self.dataframe['Rank']=self.dataframe['Rank'].map(lambda x: x[0])
        self.dataframe['Score']=self.dataframe['Score'].map(lambda x: np.round(float(x),4))
        self.dataframe['Company']=[FLAGS.retrieved_company]*len(self.dataframe)
        self.dataframe=self.dataframe.sort_values(by=['Question','Company','Model','Rank']).reset_index(drop=True)[['Question','Company','Model','Answer','Rank','Score','Doc_index','Context_Answer']]
        return self.dataframe
            

    def string_retriever(self,sentence_list):
        return [w  for w in sentence_list if not w.isdigit()]
    
    def add_ctxt(self,row):
        try:
            row['Context_Answer']=' '.join([self.contents_doc[0][1][int(row.Doc_index)-1],row.Answer,self.contents_doc[0][1][int(row.Doc_index)+1]])
        except:
            print('No context for index:',row.Doc_index)
            row['Context Answer']= ' '
        return row
    
    def update_count_dic(self,sentences,counter,is_rawtext,remove_index):
        
        for i,c in enumerate(sentences):
            counter=counter[is_rawtext].copy()
            counter[c]=counter.get(c,0)+1
            counter[is_rawtext]=counter
            if counter[is_rawtext][c]>1:
                remove_index[is_rawtext].append(i)
        return None



#######  ARGS COLLECTION TO FEED DIFFERENT MODELS ###########

def collect_args():

    if not FLAGS.demo:              
        path_q=FLAGS.query_dir
        file= open(path_q,"r+")  
        text=file.read().replace("  ","")
        queries=text.split("\n")
        queries=[q.split("+")[0] for q in queries if len(q)>1]
        topics=[q.split("+")[1].split(",") for q in queries if len(q)>1]
        file.close()
        
    else:
        queries=[FLAGS.demo_query]
        if FLAGS.model==5:
            topics=FLAGS.demo_topics
            topics=[topics.split(",")]
        
    pdf_dirs=[FLAGS.pdf_directory+FLAGS.domain+'/'+FLAGS.retrieved_company]
    grams=(FLAGS.min_gram,FLAGS.max_gram)
    
    """
    args_Infersent={'pdf_directories':pdf_dirs,'w2v_path': FLAGS.w2v_path, 'model_path': FLAGS.model_path ,'top_n':FLAGS.top_n,'ngram_range':grams,'lemmatize':FLAGS.lemmatize,'transform_text':FLAGS.transform_text,'l_questions':queries,'sentences_chunk':FLAGS.sentences_chunk,'vocab_builder':FLAGS.vocab_builder}
    args_Bert={'pdf_directories':pdf_dirs,'w2v_path': FLAGS.w2v_path, 'model_path': FLAGS.model_path ,'top_n':FLAGS.top_n,'ngram_range':grams,'lemmatize':FLAGS.lemmatize,'transform_text':FLAGS.transform_text,'l_questions':queries,'sentences_chunk':FLAGS.sentences_chunk,'vocab_builder':FLAGS.vocab_builder}
    args_Tf_Idf={'pdf_directories':pdf_dirs,'w2v_path': FLAGS.w2v_path, 'model_path': FLAGS.model_path ,'top_n':FLAGS.top_n,'ngram_range':grams,'lemmatize':FLAGS.lemmatize,'transform_text':FLAGS.transform_text,'l_questions':queries,'sentences_chunk':FLAGS.sentences_chunk,'vocab_builder':FLAGS.vocab_builder}
    args_TfBERT={'pdf_directories':pdf_dirs,'w2v_path': FLAGS.w2v_path, 'model_path': FLAGS.model_path ,'top_n':FLAGS.top_n,'ngram_range':grams,'lemmatize':FLAGS.lemmatize,'transform_text':FLAGS.transform_text,'l_questions':queries,'sentences_chunk':FLAGS.sentences_chunk,'vocab_builder':FLAGS.vocab_builder}
    args_TfBERT_enhanced={'pdf_directories':pdf_dirs,'w2v_path': FLAGS.w2v_path, 'model_path': FLAGS.model_path ,'top_n':FLAGS.top_n,'ngram_range':grams,'lemmatize':FLAGS.lemmatize,'transform_text':FLAGS.transform_text,'l_questions':queries,'sentences_chunk':FLAGS.sentences_chunk,'vocab_builder':FLAGS.vocab_builder,'topics':topics}
    """

    if FLAGS.model==1:
        return {'pdf_directories':pdf_dirs,'w2v_path': FLAGS.w2v_path, 'model_path': FLAGS.model_path ,'top_n':FLAGS.top_n,'ngram_range':grams,'lemmatize':FLAGS.lemmatize,'transform_text':FLAGS.transform_text,'l_questions':queries,'sentences_chunk':FLAGS.sentences_chunk,'vocab_builder':FLAGS.vocab_builder}
    elif FLAGS.model==2:
        return {'pdf_directories':pdf_dirs,'w2v_path': FLAGS.w2v_path, 'model_path': FLAGS.model_path ,'top_n':FLAGS.top_n,'ngram_range':grams,'lemmatize':FLAGS.lemmatize,'transform_text':FLAGS.transform_text,'l_questions':queries,'sentences_chunk':FLAGS.sentences_chunk,'vocab_builder':FLAGS.vocab_builder}
    elif FLAGS.model==3:
        return {'pdf_directories':pdf_dirs,'w2v_path': FLAGS.w2v_path, 'model_path': FLAGS.model_path ,'top_n':FLAGS.top_n,'ngram_range':grams,'lemmatize':FLAGS.lemmatize,'transform_text':FLAGS.transform_text,'l_questions':queries,'sentences_chunk':FLAGS.sentences_chunk,'vocab_builder':FLAGS.vocab_builder}
    elif FLAGS.model==4:
        return None
    elif FLAGS.model==5:
        return {'pdf_directories':pdf_dirs,'w2v_path': FLAGS.w2v_path, 'model_path': FLAGS.model_path ,'top_n':FLAGS.top_n,'ngram_range':grams,'lemmatize':FLAGS.lemmatize,'transform_text':FLAGS.transform_text,'l_questions':queries,'sentences_chunk':FLAGS.sentences_chunk,'vocab_builder':FLAGS.vocab_builder,'topics':topics}
    else:
        print('Select a correct model')


#### MAIN ######

def main(_):
    
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    dic_suffix={1:'FASTEXT',2:'BERT',3:'TFIDF',4:'BERTCLUST',5:'BERTCLUST_TUNED'}
    args=collect_args()
    check_args()
    args_fit={key:value for key,value in args.items() if key not in ['l_questions','topics']}
    QAmodel=QApipeline(**args) 
    QAmodel.fit_reader()
    QAmodel.fit()
    results=QAmodel.predict(args['l_questions'],args.get('topics',[]))
    
    if not FLAGS.demo:
        dir=FLAGS.output_dir+"/"+FLAGS.domain+"/"+FLAGS.retrieved_company
        if not os.path.exists(dir):
            os.makedirs(dir)
        results.to_csv(dir+'_result.csv')
    else:
        print('*******************************  RESULTS of BERT  ***************************************')
        show=pd.DataFrame(results[['Score','Answer']].values,index=range(0,100*(FLAGS.top_n),100),columns=['Score','Answer'])
        counter_newline={}
        for i,aw in enumerate(show.Answer.tolist()):
            dividend,quotient=len(aw)//120,len(aw)%120
            if quotient!=0:
                dividend+=1
            counter_newline[i]=dividend


            for newline in range(dividend):
                show.loc[(100*i)+(newline)]=[show.loc[100*i]['Score'],aw[120*newline:120*(newline+1)]]
        show=show.reset_index().sort_values(by='index')[['Score','Answer']].values
        indexs_tiled=np.concatenate([np.tile([i],counter_newline[i]) for i in range(FLAGS.top_n)])
        print(tabulate(pd.DataFrame(show,index=['Answer °'+ str(i) for i in indexs_tiled]), headers='keys', tablefmt='psql'))
        
tf.compat.v1.app.run()

